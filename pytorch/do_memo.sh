cd  ~/lab/gda/da/CDAN/pytorch

. cmd.sh 0 0 3_4_5_6_7_8_9_10_11_12_13_14  \
    --parent DomainNet  \
    --task contrastive_rpl_dim128_wght0.6_AE_bs512_ep2000_lr0.001_outd64_g3  \
    --method CDAN \
    --tmux CDAN
# ---------------------------------------------------------
. cmd.sh 1 0 1_2_3_4_5_6_7_8_9_10_11_12_13_14  \
    --parent DomainNet  \
    --task contrastive_rpl_dim128_wght0.6_AE_bs512_ep2000_lr0.001_outd64_g3  \
    --method CDAN+E \
    --tmux CDAN+E

