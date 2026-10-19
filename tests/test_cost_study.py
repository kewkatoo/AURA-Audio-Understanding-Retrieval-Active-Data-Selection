import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scripts import study_acquisition_cost as study
from scripts.report_acquisition_cost import frontier, full_gap_intervals


def test_preencoding_boundary_needs_only_acoustic_features_and_rejects_tampering(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    features = rng.normal(size=(20,26)).astype(np.float32)
    features /= np.linalg.norm(features,axis=1,keepdims=True)
    train = pd.DataFrame(dict(id=[str(i) for i in range(20)],label=["0","1"]*10,split="train"))
    monkeypatch.setattr(study,"BUDGETS",[10])
    monkeypatch.setattr(study,"RANDOM_BUDGETS",[10])
    monkeypatch.setattr(study,"SEEDS",[0])
    study.preselect(train,features,tmp_path,"locked")
    assert len(json.loads((tmp_path/"preencoding_locked.json").read_text())) == 2
    path = tmp_path/"preencoding/mfcc_representative_B10_seed0.json"
    saved=json.loads(path.read_text())
    assert len(set(saved["ids"])) == 10
    saved["ids"][0]="altered"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match="differs"):
        study.preselect(train,features,tmp_path,"locked")


def test_fresh_encoding_restores_order_and_verifies_reference(tmp_path,monkeypatch):
    class FakeEncoder:
        info = SimpleNamespace(embedding_dim=2)
        def encode_batch(self,waveforms,sr):
            values=np.array([[float(len(x)),float(x[0])] for x in waveforms],dtype=np.float32)
            return values/np.linalg.norm(values,axis=1,keepdims=True)
    monkeypatch.setattr(study,"process_audio",lambda path,**kwargs:SimpleNamespace(waveform=np.ones(int(path),dtype=np.float32)))
    config=study.AuraConfig()
    meta=pd.DataFrame(dict(id=["a","b","c"],audio_path=["12","5","8"]))
    reference=np.array([[12,1],[5,1],[8,1]],dtype=np.float32)
    reference/=np.linalg.norm(reference,axis=1,keepdims=True)
    output=tmp_path/"encoded.npy"
    matrix,info=study.encoded_pass(meta,FakeEncoder(),config,reference,output)
    np.testing.assert_array_equal(matrix,reference)
    assert info["max_reference_error"] == 0
    orthogonal=np.stack([-reference[:,1],reference[:,0]],axis=1)
    with pytest.raises(ValueError,match="angular"):
        study.encoded_pass(meta,FakeEncoder(),config,orthogonal,tmp_path/"wrong.npy")
    assert (tmp_path/"wrong.npy").exists()
    np.save(output,np.zeros_like(matrix))
    with pytest.raises(ValueError,match="Changed"):
        study.encoded_pass(meta,FakeEncoder(),config,reference,output)


def test_cost_frontier_preserves_accuracy_tradeoff_and_one_point_threshold():
    table=pd.DataFrame(dict(method=["full_pool","random","mfcc_representative","wav2vec_representative"],
        mean_accuracy=[.95,.94,.93,.945],total_seconds=[100,20,30,120]))
    result=frontier(table).set_index("method")
    assert result.loc["random","within_one_point_full"]
    assert not result.loc["mfcc_representative","within_one_point_full"]
    assert result.loc["mfcc_representative","dominated"]
    assert result.loc["wav2vec_representative","dominated"]
    assert not result.loc["full_pool","dominated"]


def test_full_gap_pairs_recordings_and_averages_seeds():
    rows=[]
    for method,seeds in [("full_pool",[0]),("random",[0,1])]:
        for seed in seeds:
            for i in range(4):
                rows.append(dict(method=method,budget=20 if method == "full_pool" else 10,seed=seed,
                    id=str(i),speaker_id=str(i//2),label="0",prediction="0" if method == "full_pool" else "1"))
    result=full_gap_intervals(pd.DataFrame(rows).sample(frac=1,random_state=0))
    assert result.gap_pp.iloc[0] == 100
    assert result.low_pp.iloc[0] == 100
    assert result.high_pp.iloc[0] == 100
