"""Rules declared before an independent small-corpus confirmation study."""
import math
import pandas as pd


def retention_budget(total,classes,numerator=64,denominator=175):
    if total < 1 or classes < 1 or classes > total or not 0 < numerator <= denominator:
        raise ValueError("Invalid pool, classes or retention fraction")
    return min(total,classes*math.ceil(total*numerator/(denominator*classes)))


def make_speaker_folds(records):
    frame=pd.DataFrame(records)
    required={"id","label","speaker_id","audio_hash"}
    if not required.issubset(frame) or frame[list(required)].isnull().any().any() or frame.id.duplicated().any():
        raise ValueError("Unique IDs and complete labels/speakers/hashes required")
    if frame.groupby("audio_hash").label.nunique().gt(1).any():
        raise ValueError("Identical content has conflicting labels")
    speakers=sorted(frame.speaker_id.unique())
    if len(speakers) < 3:
        raise ValueError("At least three speakers required")
    folds=[]
    for speaker in speakers:
        test=frame[frame.speaker_id == speaker]
        other=frame[frame.speaker_id != speaker]
        train=other[~other.audio_hash.isin(test.audio_hash)]
        if train.empty or set(train.label) != set(test.label):
            raise ValueError("Each fold needs the same classes in training and test")
        folds.append(dict(speaker=str(speaker),train_ids=sorted(train.id),test_ids=sorted(test.id),
            excluded_train_ids=sorted(set(other.id)-set(train.id)),
            random_budget=retention_budget(len(train),train.label.nunique())))
    return folds


def paired_order(fold_index,repeat):
    return ["full_pool","random"] if (fold_index+repeat)%2 == 0 else ["random","full_pool"]
