cd  ~/lab/gda/da/CDAN/pytorch

. cmd.sh 0 0 10  \
    --parent DomainNet  \
    --task simclr_encoder_bs512_ep2000_lr0.001_outd64_g3  \
    --method CDAN+E \
    --tmux CDAN+E__0
# ---------------------------------------------------------
. cmd.sh 1 0 13  \
    --parent DomainNet  \
    --task simclr_encoder_bs512_ep2000_lr0.001_outd64_g3  \
    --method CDAN+E \
    --tmux CDAN+E__1


# ci 0
# cp 1
# cq 2
# cr 3
# cs 4
# ip 5
# iq 6
# ir 7
# is 8
# pq 9
# pr 10
# ps 11
# qr 12
# qs 13
# rs 14
