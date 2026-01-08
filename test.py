import nemo.collections.asr as nemo_asr

# Get all model classes from the asr.models module
model_classes = [
    nemo_asr.models.ASRModel,
    nemo_asr.models.EncDecCTCModel,
    nemo_asr.models.EncDecCTCModelBPE,
    nemo_asr.models.EncDecClassificationModel,
    nemo_asr.models.EncDecFrameClassificationModel,
    nemo_asr.models.ClusteringDiarizer,
    nemo_asr.models.EncDecHybridRNNTCTCBPEModel,
    nemo_asr.models.EncDecHybridRNNTCTCModel,
    nemo_asr.models.EncDecRNNTBPEModel,
    nemo_asr.models.EncDecRNNTModel,
    nemo_asr.models.EncDecMultiTaskModel,
    nemo_asr.models.EncDecSpeakerLabelModel,
    nemo_asr.models.EncDecDiarLabelModel,
    nemo_asr.models.NeuralDiarizer,
    nemo_asr.models.SLUIntentSlotBPEModel,
    nemo_asr.models.SortformerEncLabelModel,
    nemo_asr.models.EncDecDenoiseMaskedTokenPredModel,
    nemo_asr.models.EncDecMaskedTokenPredModel,
    nemo_asr.models.SpeechEncDecSelfSupervisedModel,
    nemo_asr.models.EncDecTransfModelBPE,
    nemo_asr.models.EncDecMultiTalkerRNNTBPEModel,
]

print("=" * 80)
print("NeMo ASR Available Model Names")
print("=" * 80)

for model_class in model_classes:
    class_name = model_class.__name__
    try:
        model_names = model_class.get_available_model_names()
        print(f"\n{class_name}:")
        print("-" * 40)
        if model_names:
            for name in sorted(model_names):
                print(f"  - {name}")
        else:
            print("  (no available models)")
    except Exception as e:
        print(f"\n{class_name}:")
        print("-" * 40)
        print(f"  Error: {e}")
