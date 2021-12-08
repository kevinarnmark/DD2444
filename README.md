# DD2444
To init docker:
```
sudo docker load -i modulus_image_v21.06.tar.gz
```
To run docker container: 
```
sudo docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia -v ${PWD}/vonKarmanVortexStreet:/vonKarmanVortexStreet -it modulus:21.06 bash
```
