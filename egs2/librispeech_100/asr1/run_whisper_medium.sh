train_set="train_clean_100"
valid_set="dev"
#test_sets="test_clean test_other"
test_sets="test_clean test_other dev_clean dev_other"
asr_tag=whisper_medium_finetune_lr1e-5_adamw_wd1e-2_3epochs_guidance_newtarget
asr_config=conf/tuning/train_asr_whisper_full.yaml
inference_config=conf/decode_asr_whisper_noctc_greedy.yaml

./asr.sh \
    --skip_data_prep true \
    --skip_train false \
    --skip_eval false \
    --lang en \
    --ngpu 4 \
    --nj 4 \
    --stage 11 \
    --stop_stage 13 \
    --gpu_inference true \
    --inference_nj 1 \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model valid.acc.ave.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

#--inference_asr_model valid.acc.ave.pth \
#--cleaner whisper_en \
