#!/usr/bin/env python3

# ==========================================
#  Title:   ICE Ingestion for PostAward files
#  Author:  Prabhat Ratnala
#  Date:    16 Feb 2021
#  Version: 1.0
#  Comments: Added Error handling
# ==========================================

import os
import sys
import csv
import glob
import re
import hashlib
import itertools
import logging
import time
import traceback
from datetime import datetime
from itertools import product

import pysftp
import xlrd
from xlrd.timemachine import xrange

start_time = time.time()

# creating logging with output to log file with default logging level to info.
file_name = f"{os.getcwd()}/logs/post_award_detailed_logs_{start_time}.txt"
logging.basicConfig(filename=file_name, level=logging.INFO)

# Fetch Files from Sharepoint
# private variables for security
__sftp_host_name = "incsftp.incresearch.com"
__stfp_user_name = "INCRDC\\svc-soasftp-prd"
# __password = "hU%c63asCZeQwMpVPN!U$xdzWdfBrz"  FIX_ME: remove this post adding in command line args
__password = sys.argv[1]

cn_opts = pysftp.CnOpts()
cn_opts.hostkeys = None

try:
    with pysftp.Connection(host=__sftp_host_name, username=__stfp_user_name, password=__password,
                           cnopts=cn_opts) as sftp:
        logging.debug(f"Connection successfully established with sftp server "
                      f"- {__stfp_user_name} with username {__sftp_host_name} ...")
        logging.info(f"Connection successfully established with sftp server ...")

        remote_file_path = "/HOME/ICE/Covid19/Bulk_load_ICE"
        local_file_path = "/home/users/leap_frog_prd/ICE_Python/ICE_PostAward/Raw_Files"

        files_on_server = sftp.listdir(remote_file_path)

        for file in files_on_server:
            logging.debug(
                f"copying file - {file} from server path - {remote_file_path}/{file} to {local_file_path}/{file}")
            sftp.get(f"{remote_file_path}/{file}", f"{local_file_path}/{file}")
            logging.debug(f"removing file - {remote_file_path} from server - {remote_file_path}/{file}")
            sftp.remove(f"{remote_file_path}/{file}")

    logging.info(f"successfully copied files from {__sftp_host_name} - {remote_file_path} to {local_file_path}")
    logging.info(f"file count on stfp server - {len(files_on_server)}")

except Exception as e:
    logging.error(f"Exception while file pull from stfp server - {__sftp_host_name}...")
    logging.error(f"exception {e}.  {traceback.print_exc()}")
    sys.exit(-1)

# Read Files
input_path = "Raw_Files/"
out_path = "Raw_Files/Output/"
archive_path = "Raw_Files/Archive/"
error_path = "Raw_Files/Error/"
files = [f for f in glob.glob(input_path + "*.xls*", recursive=True)]
file_path_sheet_name = out_path + "FileSheetNames.csv"

logging.debug(f"files pulled on local - {files}")
logging.info(f"xls files count- {len(file)}")


# move file to folder
def move_file(folder, file_name):
    try:
        os.rename(file_name, folder + file_name.split("/")[1])
    except FileExistsError as f1:
        logging.warning(f"Exception for file - {file_name} - {f1}.  {traceback.print_exc()}")
        os.remove(folder + file_name.split("/")[1])
        os.rename(file_name, folder + file_name.split("/")[1])


# Column number to Alphabet
def col_num_string(n):
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string


# This will search for key value
def value_from_key(sheet, key, value):
    try:
        for row_index, col_index in product(xrange(sheet.nrows), xrange(sheet.ncols)):
            if key in str(sheet.cell(row_index, col_index).value):
                return sheet.cell(row_index, col_index + value).value
    except Exception as e1:
        logging.error(f"Unable to fetch value for cell - {key} for sheet - {sheet}")
        logging.error(f"exception - {e1}.  {traceback.print_exc()}")
        return None


# extracting file metadata for each file and appending results to sheet_name_file
def extract_file_metadata(file_obj, sheet_name_file):
    if os.path.isfile(sheet_name_file):
        logging.info(f"====== Appending data to {sheet_name_file} ======")
        file_sheet_names_file = open(sheet_name_file, "a", newline="")
    else:
        logging.info(f"====== Creating new file - {sheet_name_file} ======")
        file_sheet_names_file = open(sheet_name_file, "a", newline="")
        file_sheet_names_file.write("file_id,sheet_id,filename,project_code,template_version,sheet_name,extract_ts\n")

    book = xlrd.open_workbook(file_obj, on_demand=True)
    file_obj = file_obj.split("/")[1]
    sheet_names = book.sheet_names()
    extract_time = datetime.fromtimestamp(datetime.timestamp(datetime.now()))
    template = value_from_key(book.sheet_by_name("Executive Summary"), "Template Version", 2)
    project_code = value_from_key(book.sheet_by_name("Executive Summary"), "Project Code", 1)
    file_id = [str(int(hashlib.md5(file_obj.split("/")[1].encode("utf-8")).hexdigest(), 16))[0:7]]

    for sheet in sheet_names:
        sheet_id = [str(int(hashlib.md5(sheet.encode("utf-8")).hexdigest(), 16))[0:7]]
        filename = [file_obj]
        project_code = [project_code]
        template_version = [template]
        sheet_name = [sheet]
        extract_ts = [extract_time]
        writer = csv.writer(file_sheet_names_file)
        rows = itertools.zip_longest(file_id, sheet_id, filename, project_code, template_version, sheet_name,
                                     extract_ts,
                                     fillvalue=file_id[0])
        writer.writerows(rows)
    logging.debug(
        f"====== Data update complete for project_code - {project_code}, file_id - {file_id}, template - {template}")
    logging.info(f"====== Data update complete for file - {sheet_name_file} ======")
    book.release_resources()
    return project_code, file_id


# extracting cell data from each file
def extract_cell_data(source_loc, project_code, file_id):
    file_name = f"{out_path}{project_code}_cell_data.csv"
    logging.info(f"====== Creating new file - {file_name} ======")
    file_cell_extract = open(file_name, "w", newline="")
    file_cell_extract.write("ProjectCode,Fileid,SheetId,sheet_name,colNum,Col,RowNum,Value,ExtractTS\n")

    book = xlrd.open_workbook(source_loc, on_demand=True)
    sheet_names = book.sheet_names()
    extract_time = datetime.fromtimestamp(datetime.timestamp(datetime.now()))

    for sheet_name in sheet_names:

        sheet = book.sheet_by_name(sheet_name)
        row_lim = book.sheet_by_name(sheet_name).nrows
        col_lim = book.sheet_by_name(sheet_name).ncols
        sheet_ids = [str(int(hashlib.md5(sheet_name.encode("utf-8")).hexdigest(), 16))[0:7]]

        for row_num in range(0, row_lim):
            for col_num in range(0, col_lim):
                col_type = sheet.cell_type(row_num, col_num)
                if (not col_type == xlrd.XL_CELL_EMPTY) and (col_type != xlrd.XL_CELL_ERROR):
                    cell = str(sheet.cell_value(row_num, col_num)).replace("\n", " ").encode("utf-8")
                    if cell and not cell == " ":
                        fileid = [file_id]
                        sheet_id = [sheet_ids[0]]
                        proj_code = [project_code]
                        extract_ts = [extract_time]
                        sheet_n = [sheet_name]
                        column_number = [col_num]
                        column_string = [col_num_string(col_num + 1)]
                        row_number = [row_num + 1]
                        value = [cell]
                        writer = csv.writer(file_cell_extract)
                        rows = itertools.zip_longest(proj_code, fileid, sheet_id, sheet_n, column_number,
                                                     column_string, row_number, value, extract_ts,
                                                     fillvalue=project_code)
                        writer.writerows(rows)
    logging.info(f"====== Data upload complete for file {project_code}_cell_data.csv ======")
    book.release_resources()
    file_cell_extract.close()


# actual files processing ...
# calling extract_file_metadata and extract_cell_data
# successfully processed files moved to archive. Failed files moved to error folder
error_file_names = []
success_file_names = []
for file in files:
    logging.info(f"====== Starting ICE Budget Data Extraction for - {file} ======")
    if os.path.exists(archive_path + file.split("/")[1]):
        logging.info(f"file - {file} already ingested")
        logging.debug(f"removing file - {file} from path - {input_path}")
        os.remove(input_path + file.split("/")[1])
    else:
        try:
            file_values = extract_file_metadata(file, file_path_sheet_name)
            extract_cell_data(file, str(file_values[0][0]), str(file_values[1][0]))
            move_file(archive_path, file)
            success_file_names.append(file.split("/")[1])
            logging.info(f"====== File {file} moved to Archive ======")
        except Exception as e:
            logging.error(f"Exception while processing file {file}. {e}  {traceback.print_exc()}")
            move_file(error_path, file)
            error_file_names.append(file.split("/")[1])
            logging.error(f"====== File {file} moved to ErrorPath ======")

logging.info("====== STATS ======")
logging.info(f"====== success file names ======")
for file in success_file_names:
    logging.info(f"{file}")
logging.error(f"====== error file names ======")
for file in error_file_names:
    logging.error(f"{file}")
logging.info("Extraction completed in --- %s seconds ---" % (time.time() - start_time))
