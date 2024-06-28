import pickle


with open("lrs2_preds.pkl", 'rb') as p:
    text_dict = pickle.load(p)

pred_key = "lrs2/vid/" + "6335003747610845928/00004.npy"
raw_texts = text_dict[pred_key]['preds'][0]
print(pred_key, ": ", raw_texts)