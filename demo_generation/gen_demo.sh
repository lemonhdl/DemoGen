task=${1}
gen_range=${2}
gen_mode=${3}
n_gen_per_source=${4}
render_video=${5}

data_root=../data
# If user accidentally passes grid with 1 sample, fallback to random mode
if [ "${gen_mode}" = "grid" ] && [ "${n_gen_per_source}" = "1" ]; then
    echo "Warning: grid mode requires n_gen_per_source to be a square (and >1). Switching to 'random' mode."
    gen_mode="random"
fi

python -W ignore gen_demo.py --config-name=${task}.yaml \
                                                                data_root=${data_root} \
                                                                generation.range_name=${gen_range} \
                                                                generation.mode=${gen_mode} \
                                                                generation.n_gen_per_source=${n_gen_per_source} \
                                                                generation.render_video=${render_video}
                                     