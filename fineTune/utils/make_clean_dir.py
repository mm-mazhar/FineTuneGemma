# -*- coding: utf-8 -*-
# """
# fineTune/utils/make_clean_dir.py
# Description:
# Utility function to creaate a clean directory.
# Created on September 3, 2025
# @ Author: Mazhar
# """

import os
import shutil


def make_clean_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)  # delete entire directory
    os.makedirs(path)  # create fresh empty directory


# make_clean_dir(FINAL_ADAPTERS_OUTPUT_DIR)
# make_clean_dir(TRAINER_OUTPUT_DIR)
