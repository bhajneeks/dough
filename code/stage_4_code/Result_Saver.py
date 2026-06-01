'''
Result saver — same as Stage 3. Pickles the result dict to disk.
'''

from pathlib import Path
import pickle

from code.base_class.result import result


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')

        result_path = Path(
            self.result_destination_folder_path
            + self.result_destination_file_name
            + '_'
            + str(self.fold_count)
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)

        with open(result_path, 'wb') as f:
            pickle.dump(self.data, f)
