from tools import bleu_file, get_ref_files
import sys

if __name__ == "__main__":
	refs = get_ref_files('/data/disk1/share/zjc/nist_thulac/dev_test/nist06/nist06.en')
	result = bleu_file(sys.argv[1], refs)
	print result	
