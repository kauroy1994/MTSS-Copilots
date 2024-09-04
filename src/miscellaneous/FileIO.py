import json

class JSON:

    @staticmethod
    def read_from_path(path):
        
        f = open(path)
        json_object = json.load(f)
        f.close()

        return json_object
