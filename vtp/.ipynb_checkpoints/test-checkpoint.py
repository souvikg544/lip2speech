# from extract_emb_inf import save_visual_emb
# ar = save_visual_emb(vidpath = "/home2/souvikg544/souvik/lip2speech/vtp/data/pycrop/something/fun.avi",featpath = "/home2/souvikg544/souvik/lip2speech/vtp",ckpt_path= "/ssd_scratch/cvit/souvik/feature_extractor.pth")

# print(ar.shape)
from inference import main,run

model, video_loader, lm, lm_tokenizer = main()
pred = run('/home2/souvikg544/souvik/lip2speech/vtp/data/pycrop/something/fun.avi', video_loader, model, lm, lm_tokenizer)
print(pred)