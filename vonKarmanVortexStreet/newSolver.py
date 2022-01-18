

import os
import time
import numpy as np
import tensorflow as tf
from termcolor import colored, cprint
from copy import copy


from modulus.solver import Solver
from modulus.variables import Variables, Key
from modulus.config import ConfigObject, str2bool
from modulus.graph import unroll_graph_on_dict, unroll_graph, _needed_names
from modulus.learning_rate import LR, ExponentialDecayLR
from modulus.optimizer import AdamOptimizer
from modulus.architecture.fully_connected import FullyConnectedArch
from modulus.plot_utils.vtk import var_to_vtk, compare_var_to_vtk
from modulus.plot_utils.time_series import plot_time_series
from modulus.csv_utils.csv_rw import dict_to_csv, csv_to_dict

# import horovod if avalible
try:
  import horovod.tensorflow as hvd
except:
  # if horovod is not avalible then set rank to 1
  # NOTE there maybe be a better way of handeling this
  class horovod:
    def rank(self):
      return 0
  hvd = horovod()


class Solver2(Solver):
    convergence_check = 1.0e-5

    @classmethod
    def add_options(cls, group):
        group.add_argument('--convergence_check',
                           help='total loss convergence check',
                           type=float,
                           default=1.0e-5)

    def solve(self):
        # start tensorflow session
        self.start_session()

        # load current iteration value
        iteration_step = self.load_iteration_step()

        # make global step counter
        global_step = self.add_global_step()

        # initialize train domains
        for i, train_domain in enumerate(self.seq_train_domain):
            # add graph nodes and session to train domain for possible adaptive sampling
            # TODO this is dependency breaking and should be restructured
            train_domain.nets = self.nets
            train_domain.equations = self.equations
            train_domain.sess = self.sess
            if type(train_domain) == type:
                self.seq_train_domain[i] = train_domain(**self.config)

        # unroll graph for each train domain
        seq_train_step = []
        seq_compute_train = []
        seq_train_domain_var = []
        for i, train_domain in enumerate(self.seq_train_domain):
            # unroll graph on train domain
            train_domain_invar, train_true_domain_outvar, train_lambda_weighting = train_domain.make_inputs()
            train_pred_domain_outvar = unroll_graph_on_dict(self.nets + self.equations, train_domain_invar,
                                                            train_true_domain_outvar, diff_nodes=self.diff_nodes)

            # compute loss for domain
            losses = self.train_domain_loss(train_pred_domain_outvar, train_true_domain_outvar, train_lambda_weighting,
                                            summary_prefix='Train:' + train_domain.name)

            # add custom loss
            custom_loss = self.custom_loss(train_domain_invar, train_pred_domain_outvar, train_true_domain_outvar,
                                           global_step)
            losses.update(custom_loss)

            # compute train op
            train_op = self.initialize_optimizer(losses, global_step, self.lr.get_lr(global_step['step']))

            # train lambda functions
            train_domain_var = Variables.join_dict(train_domain_invar, train_true_domain_outvar)
            train_domain_var = Variables.join_dict(train_domain_var, train_lambda_weighting)
            train_step = Variables.lambdify_np(train_domain_var,
                                               train_op, self.sess)
            compute_train_domain_var = Variables.join_dict(train_domain_invar, train_pred_domain_outvar)
            compute_train_domain_var = Variables.join_dict(compute_train_domain_var, train_lambda_weighting)
            compute_train = Variables.lambdify_np(train_domain_var,
                                                  compute_train_domain_var, self.sess)

            # store evaluation functions
            seq_train_step.append(train_step)
            seq_compute_train.append(compute_train)
            seq_train_domain_var.append(train_domain_var)

        if self.inference_domain is not None:
            # initialize inference domain
            if type(self.inference_domain) == type:
                self.inference_domain = self.inference_domain(**self.config)

            # unroll graph on inference domain
            if self.rec_results_cpu:
                with tf.device('/cpu:0'):  # TODO clean if statement to remove nested with
                    inference_domain_invar, inference_domain_outvar_names = self.inference_domain.make_inputs()
                    inference_domain_outvar = unroll_graph_on_dict(self.nets + self.equations, inference_domain_invar,
                                                                   inference_domain_outvar_names,
                                                                   diff_nodes=self.diff_nodes)
            else:
                inference_domain_invar, inference_domain_outvar_names = self.inference_domain.make_inputs()
                inference_domain_outvar = unroll_graph_on_dict(self.nets + self.equations, inference_domain_invar,
                                                               inference_domain_outvar_names,
                                                               diff_nodes=self.diff_nodes)

            # inference lambda function
            inference_domain_var = Variables.join_dict(inference_domain_invar, inference_domain_outvar)
            compute_inference = Variables.lambdify_np(inference_domain_invar,
                                                      inference_domain_var, self.sess)

        if self.val_domain is not None:
            # initialize val domain
            if type(self.val_domain) == type:
                self.val_domain = self.val_domain(**self.config)

            # unroll graph on validation domain
            if self.rec_results_cpu:
                with tf.device('/cpu:0'):
                    val_domain_invar, val_true_domain_outvar = self.val_domain.make_inputs()
                    val_pred_domain_outvar = unroll_graph_on_dict(self.nets + self.equations, val_domain_invar,
                                                                  val_true_domain_outvar, diff_nodes=self.diff_nodes)
                    self.val_domain_error(val_pred_domain_outvar, val_true_domain_outvar, summary_prefix='val')
            else:
                val_domain_invar, val_true_domain_outvar = self.val_domain.make_inputs()
                val_pred_domain_outvar = unroll_graph_on_dict(self.nets + self.equations, val_domain_invar,
                                                              val_true_domain_outvar, diff_nodes=self.diff_nodes)
                self.val_domain_error(val_pred_domain_outvar, val_true_domain_outvar, summary_prefix='val')

            # val lambda function
            val_domain_var = Variables.join_dict(val_domain_invar, val_pred_domain_outvar)
            compute_val = Variables.lambdify_np(val_domain_invar,
                                                val_domain_var, self.sess)

        if self.monitor_domain is not None:
            # initialize monitor domain
            if type(self.monitor_domain) == type:
                self.monitor_domain = self.monitor_domain(**self.config)
            self.monitor_outvar_store = {}

            # unroll graph on monitor domain
            if self.rec_results_cpu:
                with tf.device('/cpu:0'):
                    monitor_domain_invar, monitor_domain_outvar_names, monitor_nodes = self.monitor_domain.make_inputs()
                    monitor_domain_outvar = unroll_graph_on_dict(self.nets + self.equations + monitor_nodes,
                                                                 monitor_domain_invar, monitor_domain_outvar_names,
                                                                 diff_nodes=self.diff_nodes)
                    for name, m in monitor_domain_outvar.items():
                        Variables.tf_summary(m, prefix='monitor/' + name)
            else:
                monitor_domain_invar, monitor_domain_outvar_names, monitor_nodes = self.monitor_domain.make_inputs()
                monitor_domain_outvar = unroll_graph_on_dict(self.nets + self.equations + monitor_nodes,
                                                             monitor_domain_invar, monitor_domain_outvar_names,
                                                             diff_nodes=self.diff_nodes)
                for name, m in monitor_domain_outvar.items():
                    Variables.tf_summary(m, prefix='monitor/' + name)

            # monitor lambda function
            compute_monitor = Variables.lambdify_np(monitor_domain_invar,
                                                    monitor_domain_outvar, self.sess)

        # make summary writer
        self.summary_op = Variables({Key('summary_str'): tf.summary.merge_all()})
        #  self.summary_writer = tf.summary.FileWriter(self.network_dir)

        # summary lambda function
        summary_var = {}
        for train_domain_var in seq_train_domain_var:
            summary_var.update(train_domain_var)
        if self.val_domain is not None:
            summary_var.update(Variables.join_dict(val_domain_invar, val_true_domain_outvar))
        if self.monitor_domain is not None:
            summary_var.update(monitor_domain_invar)
        get_summary_str = Variables.lambdify_np(summary_var,
                                                self.summary_op, self.sess)

        # make user given custom update op
        custom_update_op = self.custom_update_op()

        # load current domain index and iteration index and restore network
        self.run_initialization()
        current_domain_index, current_iteration_index = self.load_iteration_step()
        self.load_network(current_domain_index, current_iteration_index)
        self.broadcast_initialization()

        # solve for each domain in seq_train_domin
        for domain_index in range(current_domain_index, len(self.seq_train_domain)):
            # solve for number of iterations in train_domain
            for iteration_index in range(current_iteration_index, self.seq_train_domain[domain_index].nr_iterations):
                # make domain dirs
                self.make_domain_dirs(domain_index,
                                      iteration_index)

                # make neural network saver
                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

                # save current iteration step
                self.save_iteration_step(domain_index,
                                         iteration_index)

                # make summary writer
                if hvd.rank() == 0:
                    summary_writer = tf.summary.FileWriter(self.base_dir(domain_index, iteration_index))

                # solve for domain
                print("Solving for Domain " + str(self.seq_train_domain[domain_index].name) + ' iteration ' + str(
                    iteration_index))
                t = time.time()

                first = True
                initial_loss = np.float32(0.0)  # Used to calc relative loss used in early termination

                while True:
                    train_np_var = self.seq_train_domain[domain_index].sample()
                    train_stats = seq_train_step[domain_index](train_np_var)

                    #total_loss = train_stats['total_loss']
                    if first: # Get initial total loss for use in relative loss for early termination
                        first = False
                        initial_loss = train_stats['total_loss']
                        #print("Initial total loss: " + str(initial_loss))

                    # check for nans in loss
                    if (hvd.rank() == 0):
                        if np.isnan(np.sum([value for key, value in train_stats.items() if 'loss' in key.name])):
                            print("loss went to Nans")
                            break

                    if (hvd.rank() == 0) and (train_stats['step'] % self.rec_results_freq == 0) and self.rec_results:
                        # record train batch
                        train_true_np_var = self.seq_train_domain[domain_index].sample()
                        self.record_train(train_true_np_var,
                                          seq_compute_train[domain_index](train_true_np_var),
                                          train_stats['step'])

                        # record inference batch
                        if self.inference_domain is not None:
                            inference_np_var = self.inference_domain.sample()
                            self.record_inference(compute_inference(inference_np_var),
                                                  train_stats['step'])

                        # record val batch
                        if self.val_domain is not None:
                            val_np_var = self.val_domain.sample()
                            self.record_validation(val_np_var,
                                                   compute_val(val_np_var),
                                                   train_stats['step'])

                        # record monitor
                        if self.monitor_domain is not None:
                            monitor_np_var = self.monitor_domain.sample()
                            self.record_monitor(compute_monitor(monitor_np_var),
                                                train_stats['step'])

                    if (hvd.rank() == 0) and (train_stats['step'] % self.print_stats_freq == 0):
                        for key in train_stats.keys():
                            if 'loss' in key.name:  # print values with loss
                                print(key.name + ": " + str(train_stats[key]))
                        elapsed = (time.time() - t) / self.print_stats_freq
                        t = time.time()
                        print("time: " + str(elapsed))

                    if (hvd.rank() == 0) and (train_stats['step'] % self.tf_summary_freq == 0):
                        np_var = {}
                        for train_domain in self.seq_train_domain:
                            np_var.update(train_domain.sample())
                        if self.val_domain is not None:
                            np_var.update(self.val_domain.sample())
                        if self.monitor_domain is not None:
                            np_var.update(self.monitor_domain.sample())
                        summary_str = get_summary_str(np_var)['summary_str']
                        summary_writer.add_summary(summary_str, train_stats['step'])

                    if (hvd.rank() == 0) and (train_stats['step'] % self.save_network_freq == 0):
                        self.save_checkpoint(global_step=train_stats['step'],
                                             domain_index=domain_index,
                                             iteration_index=iteration_index)
                        print("saved to " + self.base_dir(domain_index, iteration_index))

                    if (abs(train_stats['total_loss']/initial_loss)) < self.convergence_check:
                        if hvd.rank() == 0:
                            print("Debug: " + str(train_stats['total_loss']) + " / " + str(initial_loss))
                            print("Finished training! Relative loss of "
                                  + str(train_stats['total_loss']/initial_loss)
                                  + " has been reached")
                        break

                    # if (train_stats['step'] >= self.max_steps) or (train_stats['total_loss'] < self.convergence_check):
                    if train_stats['step'] >= self.max_steps:
                        if hvd.rank() == 0:
                            print("Finished training! Max number of steps reached")
                        break

                # run user given operation to update weights
                if custom_update_op is not None:
                    self.sess.run(custom_update_op)
