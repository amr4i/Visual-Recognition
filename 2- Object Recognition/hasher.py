import sys,os
import hashlib

def md5(fname):
	hash_md5 = hashlib.md5()
	with open(fname, "rb") as f:
		for chunk in iter(lambda: f.read(4096), b""):
			hash_md5.update(chunk)
	f.close()

	# with open(os.path.join(out_path, file_path.split("/")[-1].split(".")[0]+".md5"), 'w') as g:
	#   g.write(hash_md5.hexdigest())
	# g.close()

	return hash_md5.hexdigest()
	

def hash_models(model_folder):
	f = open("output_hash.txt", "w")
	for file in os.listdir(model_folder):
		file_path = os.path.join(model_folder, file)
		file_hash = md5(file_path)
		f.write(file+" "+file_hash+"\n")
	f.close()

if __name__ == '__main__':
	model_folder = 'models/final'
	hash_models(model_folder)

