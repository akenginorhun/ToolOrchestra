docker run --gpus '"device=0,1"' -it --rm --shm-size=128g \
	      --env USER_ID=$(id -u) --env GROUP_ID=$(id -g)  \
	            -v $PWD:/openrlhf -v $PWD/../../:/home/myuser -v $PWD/../data/:/datasets -v $PWD/../results/:/results -v $PWD/../:/Projects --ulimit memlock=-1 --ulimit \
		                    stack=67108864 nvcr.io/nvidian/nemo:verl_v2
