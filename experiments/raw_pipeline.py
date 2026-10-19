"""Uncached raw-audio encoding for prospective acquisition experiments."""
import numpy as np
from aura.ingest.audio import process_audio


def encode_recordings(metadata, encoder, audio, chunk_size=500):
    if chunk_size < 1 or metadata.empty or metadata.id.duplicated().any():
        raise ValueError("Nonempty unique recording IDs and a positive chunk size are required")
    arrays=[]
    for start in range(0,len(metadata),chunk_size):
        frame=metadata.iloc[start:start+chunk_size]
        waves=[process_audio(path,target_sample_rate=audio.sample_rate,
            do_mono=audio.mono,do_normalize=audio.normalize,do_vad=audio.vad,do_trim=audio.vad,
            vad_frame_ms=audio.vad_frame_ms,vad_hop_ms=audio.vad_hop_ms,
            vad_energy_percentile=audio.vad_energy_percentile,
            vad_noise_margin_db=audio.vad_noise_margin_db,vad_min_rms=audio.vad_min_rms).waveform
            for path in frame.audio_path]
        order=sorted(range(len(waves)),key=lambda i:len(waves[i]))
        matrix=np.empty((len(frame),encoder.info.embedding_dim),dtype=np.float32)
        for offset in range(0,len(order),encoder.batch_size):
            batch=order[offset:offset+encoder.batch_size]
            matrix[batch]=encoder.encode_batch([waves[i] for i in batch],audio.sample_rate)
        arrays.append(matrix)
    return np.concatenate(arrays)
