import sys,os
import hashlib

def md5(fname, out_path):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    f.close()

    with open(os.path.join(out_path, file_path.split("/")[-1].split(".")[0]+".md5"), 'w') as g:
    	g.write(hash_md5.hexdigest())
   	g.close()

    # return hash_md5.hexdigest()
	


if __name__ == '__main__':
	file_path = sys.argv[1]
	out_path = sys.argv[2]
	md5(file_path, out_path)

