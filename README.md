# DD2444
To init docker:
```
sudo docker load -i modulus_image_v21.06.tar.gz
```
To run docker container simulating the vonKarmanVortexStreet: 
```
sudo docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia -v ${PWD}/vonKarmanVortexStreet:/vonKarmanVortexStreet -it modulus:21.06 bash
```
To run docker container simulating the poisson problem: 
```
sudo docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia -v ${PWD}/poisson:/poisson -it modulus:21.06 bash
```

To run the FEniCS DOLFIN simulations use the following in either the poisson or the vonKarmanVortexStreet directory:
```
sudo docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable:current
```
