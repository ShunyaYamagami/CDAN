. cmd.sh 0 0 5   \
    -p OfficeHome  \
    --method CDAN  \
    --task   simclr_bs512_ep1000_g3_shfl  \
    --tmux   CDANE_0

. cmd.sh 1 0 5 \
    -p OfficeHome  \
    --method CDAN+E  \
    --task   simclr_bs512_ep1000_g3_shfl  \
    --tmux   CDANE_1

