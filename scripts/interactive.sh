if [ -f startup.sh ];
then
    source startup.sh;
else
    source scripts/startup.sh;
fi
srun --gpus=1 -n 16 --mem-per-cpu=4096 --pty bash
