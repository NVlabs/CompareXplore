# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from utils import get_log_path, get_file_path, create_dir_if_not_exists, load_pickle
import os
from os.path import join, dirname, basename, exists
from zipfile import ZipFile
from glob import glob

FOLDER_LI = \
'''


in_ds_2024-01-20T02-10-01.427611_r_Scai1_yba
in_ds_2024-01-20T17-51-48.369353_r_Scai1_yba
in_ds_2024-01-20T17-52-06.778570_r_Scai1_yba
in_ds_2024-01-21T15-02-15.384491_r_Scai1_yba
in_ds_2024-01-20T17-54-04.789823_r_Scai1_yba
in_ds_2024-01-20T17-54-47.985707_r_Scai1_yba

'''

def main():
    folder_li = FOLDER_LI.strip().split()
    files_to_zip = []
    model_name = None
    for folder in folder_li:
        # print(folder)
        glob_path = join(get_log_path(), folder, 'dse*', '*.pkl')
        # print(glob_path)
        file_li = glob(glob_path)
        # print(file_li)
        if len(file_li) == 0:
            raise RuntimeError(f'Folder {folder} does not contain any DSE result pkl files!')
        for file in file_li:
            parsed_model_name = basename(dirname(file))
            if model_name is None:
                model_name = parsed_model_name
            else:
                if model_name != parsed_model_name:
                    raise RuntimeError(f'model_name parsed = {parsed_model_name} from {file} different from a previously parsed model name {model_name}')
        files_to_zip.extend(file_li)
    if len(files_to_zip) == 0:
        return
    
    assert model_name is not None
 

    # Create a ZipFile Object
    zip_file_name = join(get_file_path(), 'dse', f'{model_name}.zip')
    create_dir_if_not_exists(dirname(zip_file_name))
    with ZipFile(zip_file_name, 'w') as zip_object:
        for file in files_to_zip:
            print(f'Adding {file} to zip')
            zip_object.write(file)
    
    # Check to see if the zip file is created
    if exists(zip_file_name):
        print(f"{zip_file_name} created in\n{dirname(zip_file_name)}")
    else:
        print(f"ZIP file {zip_file_name} not created")

if __name__ == '__main__':
    main()

