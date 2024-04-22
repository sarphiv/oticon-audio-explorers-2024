@echo off

REM ***** Datasets for ICASSP 2022 DNS Challenge 4 - Personalized DNS Track *****

REM NOTE: Before downloading, make sure you have enough space
REM on your local storage!

REM In all, you will need about 380TB to store the UNPACKED data.
REM Archived, the same data takes about 200GB total.

REM Please comment out the files you don't need before launching
REM the script.

REM NOTE: By default, the script *DOES NOT* DOWNLOAD ANY FILES!
REM Please scroll down and edit this script to pick the
REM downloading method that works best for you.

REM -------------------------------------------------------------
REM The directory structure of the unpacked data is:

REM . 362G
REM +-- datasets_fullband 64G
REM |   +-- impulse_responses 5.9G
REM |   \-- noise_fullband 58G
REM +-- pdns_training_set 294G
REM |   +-- enrollment_embeddings 115M
REM |   +-- enrollment_wav 42G
REM |   +-- raw/clean 252G
REM |       +-- english 168G
REM |       +-- french 2.1G
REM |       +-- german 53G
REM |       +-- italian 17G
REM |       +-- russian 6.8G
REM |       \-- spanish 5.4G
REM \-- personalized_dev_testset 3.3G

setlocal enabledelayedexpansion

REM set "BLOB_NAMES=pdns_training_set/raw/pdns_training_set.raw.clean.english_000.tar.bz2 
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_001.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_002.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_003.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_004.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_005.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_006.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_007.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_008.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_009.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_010.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_011.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_012.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_013.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_014.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_015.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_016.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_017.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_018.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_019.tar.bz2
REM                 pdns_training_set/raw/pdns_training_set.raw.clean.english_020.tar.bz2
REM                 pdns_training_set/enrollment_wav/pdns_training_set.enrollment_wav.english_000.tar.bz2
REM                 pdns_training_set/enrollment_wav/pdns_training_set.enrollment_wav.english_001.tar.bz2
REM                 pdns_training_set/enrollment_wav/pdns_training_set.enrollment_wav.english_002.tar.bz2
REM                 pdns_training_set/enrollment_wav/pdns_training_set.enrollment_wav.english_003.tar.bz2
REM                 pdns_training_set/enrollment_wav/pdns_training_set.enrollment_wav.english_004.tar.bz2
REM                 pdns_training_set/pdns_training_set.enrollment_embeddings_000.tar.bz2
REM                 datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2
REM                 datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2
REM                 datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_002.tar.bz2
REM                 datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_003.tar.bz2
REM                 datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_004.tar.bz2
REM                 datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_005.tar.bz2
REM                 datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_006.tar.bz2
REM                 datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2
REM                 datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2
REM                 datasets_fullband/datasets_fullband.impulse_responses_000.tar.bz2
REM                 personalized_dev_testset/personalized_dev_testset.enrollment.tar.bz2
REM                 personalized_dev_testset/personalized_dev_testset.noisy_testclips.tar.bz2" 

set "BLOB_NAMES=datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_003.tar.bz2 datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2 datasets_fullband/noise_fullband/datasets_fullband.noise_fullband.audioset_006.tar.bz2


set "AZURE_URL=https://dns4public.blob.core.windows.net/dns4archive"
set "OUTPUT_PATH=."

mkdir "%OUTPUT_PATH%\pdns_training_set\raw"
mkdir "%OUTPUT_PATH%\pdns_training_set\enrollment_wav"
mkdir "%OUTPUT_PATH%\datasets_fullband\noise_fullband"

for %%B in (%BLOB_NAMES%) do (
    set "BLOB=%%B"
    set "URL=%AZURE_URL%/!BLOB!"
    echo Download: !BLOB!

    REM DRY RUN: print HTTP response and Content-Length
    REM WITHOUT downloading the files
    curl.exe -s -I "!URL!" | more +2

    REM Actually download the files: UNCOMMENT when ready to download
    curl.exe "!URL!" -o "%OUTPUT_PATH%\!BLOB!"

    REM Same as above, but using wget
    REM wget.exe "!URL!" -O "%OUTPUT_PATH%\!BLOB!"

    REM Same, + unpack files on the fly
    REM curl.exe "!URL!" | tar.exe -C "%OUTPUT_PATH%" -f - -x -j
)
