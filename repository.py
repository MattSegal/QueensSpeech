import cPickle as pickle
import bsddb
import sys

class Repository:
	"""
	This thing will save Python objects into the database file provided.
	Just rename an empty text file to 'blah.db' and you're good to go.
	It will save and retrieve whole objects, which is pretty sweet.
	"""
	def __init__(self,database_file):
		self.db = bsddb.hashopen(database_file,'c')

	def save_object(self,obj,key):
		self.db[key] = pickle.dumps(obj)

	def save_without_overwrite(self,obj,key):
		raise NotImplementedError()

	def delete_key(self,key):
		del self.db[key]

	def load_object(self,key):
		return pickle.loads(self.db[key])

	def check_for_key(self,key):
		return self.db.has_key(key)

	def get_keys(self):
		return self.db.keys()

if __name__ == "__main__":
	db_file = sys.argv[1]
	repo = Repository(db_file)
	print "Database keys:"
	for key in repo.get_keys():
		print "\t" + key
	print ""