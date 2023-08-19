cd  ~/lab/gda/da/CDAN/pytorch

. cmd.sh 0 0 -1   \
    -p OfficeHome  \
    --method CDAN+E  \
    --task   contrastive_rpl_dim128_wght0.6_AE_bs512_ep3000_outd64_g3  \
    --tmux   CDAN+E_0
