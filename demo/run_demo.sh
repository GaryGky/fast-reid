python demo/visualize_result.py --config-file configs/VeRi/sbs_R50-ibn.yml \
--parallel --vis-label --dataset-name 'VeRi' --output logs/veri \
--opts MODEL.WEIGHTS logs/veri/sbs_R50-ibn/model_final.pth




python demo/visualize_result.py --config-file configs/VeRi/sbs_R50-ibn.yml \
--parallel --vis-label --dataset-name 'VeRi' --output logs/veri/resnest \
--opts MODEL.WEIGHTS logs/veri/resnest/sbs_Rs50-ibn/model_final.pth
