#!/bin/bash
#
# Simple SLURM script for submitting multiple serial
# jobs (e.g. parametric studies) using a script wrapper
# to launch the jobs.
#
# To use, build the launcher executable and your
# serial application(s) and place them in your WORKDIR
# directory.  Then, edit the CONTROL_FILE to specify 
# each executable per process.
#-------------------------------------------------------
#-------------------------------------------------------
# 
#         <------ Setup Parameters ------>
#
#SBATCH -J ensemble
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -p skx-normal
#SBATCH -t 12:00:00
#SBATCH -A TG-MCB180164
#------------------------------------------------------

module load launcher
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=/scratch/02129/jayab867/systemhc/ensemble/jobfile
 
$LAUNCHER_DIR/paramrun

