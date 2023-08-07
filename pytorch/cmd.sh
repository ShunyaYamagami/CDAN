function process_args {
    declare -A args

    # 無名引数数
    local gpu_i=$1
    local exec_num=$2
    local dset_num=$3  # -1の時, 前dsetを実行
    shift 3  # 無名引数数

    # 残りの名前付き引数を解析
    local parent="OfficeHome"
    local method="CDAN"
    local tmux_session=""
    
    local params=$(getopt -n "$0" -o p:t: -l parent:,task:,method:,tmux: -- "$@")
    eval set -- "$params"

    while true; do
        case "$1" in
            -p|--parent)
                parent="$2"
                shift 2
                ;;
            -t|--task)
                task=("$2")
                shift 2
                ;;
            --method)
                method="$2"
                shift 2
                ;;
            --tmux)
                tmux_session="$2"
                shift 2
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "不明な引数: $1" >&2
                return 1
                ;;
        esac
    done


    if [ $parent = 'Office31' ]; then
        local task=(
            # "original_uda"
            "true_domains"
            "simclr_rpl_uniform_dim512_wght0.5_bs512_ep300_g3_encoder_outdim64_shfl"
            "simclr_bs512_ep300_g3_shfl"
            "simple_bs512_ep300_g3_AE_outd64_shfl"
            "contrastive_rpl_dim512_wght0.6_AE_bs256_ep300_outd64_g3"
        )
    elif [ $parent = 'OfficeHome' ]; then
        local task=(
            # "original_uda"
            "true_domains"
            "simclr_rpl_dim128_wght0.5_bs512_ep3000_g3_encoder_outdim64_shfl"
            # "simclr_bs512_ep1000_g3_shfl"
        )
    fi
    
    
    echo "gpu_i: $gpu_i"
    echo "exec_num: $exec_num"
    echo "dset_num: $dset_num"
    echo "parent: $parent"
    echo "method: $method"
    echo -e ''  # (今は使っていないが)改行文字は echo コマンドに -e オプションを付けて実行した場合にのみ機能する.
    
    ##### データセット設定
    if [ $parent = 'Office31' ]; then
        # dsetlist=("amazon_dslr" "webcam_amazon" "dslr_webcam")
        dsetlist=("amazon_webcam" "webcam_dslr" "dslr_amazon")
        # dsetlist=("amazon_dslr" "amazon_webcam" "webcam_dslr" "webcam_amazon" "dslr_amazon" "dslr_webcam")
    elif [ $parent = 'OfficeHome' ]; then
        # dsetlist=("Art_Clipart" "Art_Product" "Art_RealWorld" "Clipart_Product" "Clipart_RealWorld" "Product_RealWorld")
        dsetlist=('Clipart_Art' 'Product_Art' 'Product_Clipart' 'RealWorld_Art' 'RealWorld_Clipart' 'RealWorld_Product')
        # dsetlist=('Art_Clipart' 'Art_Product' 'Art_RealWorld' 'Clipart_Art' 'Clipart_Product' 'Clipart_RealWorld' 'Product_Art' 'Product_Clipart' 'Product_RealWorld' 'RealWorld_Art' 'RealWorld_Clipart' 'RealWorld_Product')
    elif [ $parent = 'DomainNet' ]; then
        dsetlist=('clipart_infograph' 'clipart_painting' 'clipart_quickdraw' 'clipart_real' 'clipart_sketch' 'infograph_painting' 'infograph_quickdraw' 'infograph_real' 'infograph_sketch' 'painting_quickdraw' 'painting_real' 'painting_sketch' 'quickdraw_real' 'quickdraw_sketch' 'real_sketch')
    else
        echo "不明なデータセット: $parent" >&2
        return 1
    fi
    
    COMMAND="conda deactivate && conda deactivate"
    COMMAND+=" && conda activate cdan"
    
    local test_interval=500
    
    for tsk in "${task[@]}"; do
        if [ $dset_num -eq -1 ]; then
            for dset in "${dsetlist[@]}"; do
                COMMAND+=" && python train_image.py  "$method"  --gpu_id $gpu_i  --net ResNet50  --dataset $parent  --dset $dset  --task $tsk  --test_interval $test_interval"
            done
        else
            dset=${dsetlist[$dset_num]}
            COMMAND+=" && python train_image.py  "$method"  --gpu_id $gpu_i  --net ResNet50  --dataset $parent  --dset $dset  --task $tsk  --test_interval $test_interval"
        fi
    done

    echo $COMMAND
    echo ''
    if [ -n "$tmux_session" ]; then
        # 第3引数が存在する場合の処理. tmux内で実行する. $tmux_sessionはtmuxのセッション名.
        tmux -2 new -d -s $tmux_session
        tmux send-key -t $tmux_session.0 "$COMMAND" ENTER
    else
        # 第3引数が存在しない場合の処理. そのまま実行.
        eval $COMMAND
    fi
}


# 最初の3つの引数をチェック
if [ "$#" -lt 3 ]; then
    echo "エラー: 引数が足りません。最初の3つの引数は必須です。" >&2
    return 1
fi

########## Main ##########
process_args "$@"