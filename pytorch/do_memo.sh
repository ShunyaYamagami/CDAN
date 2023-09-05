cd  ~/lab/gda/da/CDAN/pytorch

. cmd.sh 0 0 1_2_3_4_5_6_7     --parent DomainNet  --task true_domains  --method CDAN+E  --tmux DomainNet_0

. cmd.sh 1 0 14  --parent DomainNet  --task true_domains  --method CDAN+E  --resume CDAN+E/DomainNet/230904_19:32:48--c1n0--real_sketch--true_domains  --tmux DomainNet_1
. cmd.sh 1 0 13_12_11_10_9_8  --parent DomainNet  --task true_domains  --method CDAN+E  --tmux DomainNet_1
