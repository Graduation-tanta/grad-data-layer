import _sqlite3

db_name = input("DataBase Name: ")
con = _sqlite3.connect('' + db_name)


class DocDB(object):

    # function to Select 10000 documents and next 10000 docs,...etc

    def retrieval(self):
        cur = con.cursor()

        # the number of documents that will be read from DataSet
        batchsize = 500
        offset = 0  # offset of the first document in next batch

        while True:
            # query that will select 10000 from DataSet and then 10000 ..etc
            cur.execute('SELECT * FROM documents LIMIT ? OFFSET ?', (batchsize, offset))

            # where returned documents are stored
            batch = list(cur)

            offset += batchsize
            if not batch:
                break

            for num in batch:
                yield num

    # function to get text of documents from DataSet by using id
    def get_text(self, id):
        cur = con.cursor()
        # query select text under where condition of id
        cur.execute("SELECT text FROM documents WHERE id =?", (id,))

        # take result coming from select query and stored them in array
        results = [r[0] for r in cur.fetchall()]
        # close connection with DataSet
        cur.close()
        return results












