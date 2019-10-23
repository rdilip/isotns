import pickle
def tmpsave(obj):
    with open("tmp.pkl", "wb+") as f:
        pickle.dump(obj, f)

