

# Send a ping to confirm a successful connection
try:
  client.admin.command('ping')
  print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
  print(e)

student = {
  'name': 'bob',
  'grade': 12,
  'age': 13,
  'parents' : ['Mom', 'Dad'],
  'scores' : [
    90, 92, 93
  ]
}

db.admin.insert_many([student, student, student])


#db.admin.update_one({}, {'$set': {'number': 1}}, upsert=True)
#db.admin.delete_one({'grade':10})
client.close()
