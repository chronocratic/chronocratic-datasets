#!/usr/bin/env bash
#
# hpc_config.sh — HPC cluster configuration (Kathleen cluster).
#
# Copy this file to hpc_config.sh and set the variables below.
# Do NOT commit hpc_config.sh (it is gitignored).
#

# Single source of truth for HPC storage path (required).
# All experiment outputs go under this directory.
HPC_OUTPUT_ROOT="/storage/work/skaf/outputs/rbs_paper"

# SLURM partition
HPC_PARTITION="Kathleen"

# SLURM account
HPC_ACCOUNT=""

# Time limit (SLURM format: D-HH:MM:SS)
HPC_TIME="2-00:00:00"

# Memory per CPU
HPC_MEM_PER_CPU="16G"

# CPUs per task
HPC_CPUS_PER_TASK=4

# Maximum concurrent array tasks
HPC_MAX_CONCURRENT=128

# QoS level
HPC_QOS="medium"

# Delete job files after submission (true/false)
DELETE_JOB_FILES=false
