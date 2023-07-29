function process_args {
    declare -A args

    # 無名引数数
    local gpu_i=$1
    local exec_num=$2
    local dset_num=$3  # -1の時, 前dsetを実行
    shift 3  # 無名引数数

    # 残りの名前付き引数を解析
    local parent="Office31"

    if [ $parent = 'Office31' ]; then
        local task=(
            # "original_uda"
            # "true_domains"
            # "simclr_rpl_uniform_dim512_wght0.5_bs512_ep300_g3_encoder_outdim64_shfl"
            # "simclr_bs512_ep300_g3_shfl"
            "simple_bs512_ep300_g3_AE_outd64_shfl"
            "contrastive_rpl_dim512_wght0.6_AE_bs256_ep300_outd64_g3"
        )
    elif [ $parent = 'OfficeHome' ]; then
        local task=(
            # "original_uda"
            # "true_domains"
            "simclr_rpl_dim128_wght0.5_bs512_ep3000_g3_encoder_outdim64_shfl"
            "simclr_bs512_ep1000_g3_shfl"
        )
    fi
    
    local method="CDAN+E"
    
    local params=$(getopt -n "$0" -o p:t: -l parent:,task:,method: -- "$@")
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
    echo "gpu_i: $gpu_i"
    echo "exec_num: $exec_num"
    echo "dset_num: $dset_num"
    echo "parent: $parent"
    echo "method: $method"
    echo -e ''  # (今は使っていないが)改行文字は echo コマンドに -e オプションを付けて実行した場合にのみ機能する.
    
    ##### データセット設定
    if [ $parent = 'Office31' ]; then
        dsetlist=("amazon_dslr" "dslr_webcam" "webcam_amazon")
    elif [ $parent = 'OfficeHome' ]; then
        dsetlist=("Art_Clipart" "Art_Product" "Art_RealWorld" "Clipart_Product" "Clipart_RealWorld" "Product_RealWorld")
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
    eval $COMMAND
}


# 最初の3つの引数をチェック
if [ "$#" -lt 3 ]; then
    echo "エラー: 引数が足りません。最初の3つの引数は必須です。" >&2
    return 1
fi

########## Main ##########
process_args "$@"