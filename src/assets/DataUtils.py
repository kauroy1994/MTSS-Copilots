class DataLoader:

    @staticmethod
    def read_data():

        f = open('/teamspace/studios/this_studio/MTSS-Copilots/src/assets/Final_txt_document_course.txt')
        f_lines = f.read().splitlines()
        f.close()
        return f_lines