import nemo.collections.asr as nemo_asr

model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
    "./parakeet-rnnt-1.1b-multilingual.nemo"
)
model.eval()
model = model.cuda()

audio_files = [
    "test_audios/test-1.wav",
    "test_audios/test-2.wav",
    "test_audios/test-3.wav",
    "test_audios/test-4.wav",
    "test_audios/test-5.wav",
    "test_audios/test-6.wav",
    "test_audios/test-7.wav",
    "test_audios/test-8.wav",
    "test_audios/test-9.wav",
    "test_audios/test-10.wav",
]

transcriptions = model.transcribe(audio=audio_files, batch_size=4)

for f, t in zip(audio_files, transcriptions):
    print(f"{f}: {t.text}")
