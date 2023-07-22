function process_args {
    declare -A args

    # 無名引数数
    local gpu_i=$1
    local exec_num=$2
    local dset_num=$3  # -1の時, 前dsetを実行
    shift 3  # 無名引数数

    # 残りの名前付き引数を解析
    local parent="office"  # データセット名のデフォルト値
    local task="original_uda"
    # local task="true_domains"
    # local task="simclr_rpl_uniform_dim512_wght0.5_bs512_ep300_g3_encoder_outdim64_shfl"
    # local task="simclr_bs512_ep300_g3_shfl"
    
    local params=$(getopt -n "$0" -o p:t: -l parent:,task: -- "$@")
    eval set -- "$params"

    while true; do
        case "$1" in
            -p|--parent)
                parent="$2"
                shift 2
                ;;
            -t|--task)
                parent="$2"
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
    echo -e ''  # (今は使っていないが)改行文字は echo コマンドに -e オプションを付けて実行した場合にのみ機能する.
    
    python train_image.py \
        'CDAN+E' \
        --gpu_id $gpu_i \
        --net ResNet50 \
        --dset $parent \
        --dset_num $dset_num \
        --task $task \
        --test_interval 500
}


# 最初の3つの引数をチェック
if [ "$#" -lt 3 ]; then
    echo "エラー: 引数が足りません。最初の3つの引数は必須です。" >&2
    return 1
fi

########## Main ##########
process_args "$@"