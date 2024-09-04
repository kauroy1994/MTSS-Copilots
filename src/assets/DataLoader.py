import os
from miscellaneous.FileIO import JSON

class MTSS_data_loader:

    roles = ['admin','clinical_staff','school_counselor','school_psych','teachers']

    def get_data_from_folder_path(data_folder_path, role = None):

        if role is None:

            all_qa = list()

            partition_file_names = os.listdir(os.getcwd()+data_folder_path)
            for partition_file_name in partition_file_names:
                file_path = os.path.join(os.getcwd()+data_folder_path,partition_file_name)
                file_qa = None
                try:
                    file_qa = list(JSON.read_from_path(file_path))

                except Exception as error:
                    continue

                all_qa += file_qa

            return all_qa

        else:

            qa = list()
            qa_file_path = os.getcwd()+data_folder_path.replace("/json_files","/")+role+'_qa.json'
            qa = list(JSON.read_from_path(qa_file_path))
            return qa

