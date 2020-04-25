import sys
import os
from subprocess import call
import glob

#receptor_name = "REDOCK"
#additional_flags = "-n $cores -p"
receptor_name = "REPLACE_RECEPTOR"
additional_flags = "-n 6 --clean_rcd --no_progress -a 4"
num_replicates = 1
num_runs_per_script = 406 # number of systems (will run for number of replicates each)
folder_location = "/scratch/02129/jayab867/nonbinders_netmhcpan/stampede_scripts/"
#hla_location = "/scratch/02129/jayab867/systemhc/hlas/"
hla_location = "/scratch/02129/jayab867/nonbinders_netmhcpan/hlas/"
# for davinci, num_replicates * num_runs_per_script should be around 64

f = open(sys.argv[1], 'r')
pHLAs = []
for line in f:
    hla, peptide = line.split()
    pHLAs.append((hla, peptide))
f.close()

counter = 0
num_scripts = 0
call(["cp run_sampling_long.sh run_sampling_long_" + str(num_scripts) + ".sh"], shell=True)
f = open("run_sampling_long_" + str(num_scripts) + ".sh", 'a')
#call(["sed -i \"s/test/test-" + str(num_scripts) + "/g\" run_sampling_long_" + str(num_scripts) + ".sh"], shell=True)

for i, p in enumerate(pHLAs):

    hla, sequence = p
    #hla_name = "HLA-" + hla[0] + "*" + hla[1:3] + ":" + hla[3:5]

    #f = open("temp.txt", 'w')
    #f = open("run_sampling_long_" + str(num_scripts) + ".sh", 'a')
    f.write("cd /dev/shm\n")
    f.write("mkdir -p " + hla + "/" + sequence + "\n")
    f.write("cd " + hla + "/" + sequence + "\n")

    if True: #os.path.exists(hla_location + hla + ".pdb"):
      f.write("cp " + hla_location + hla + ".pdb .; singularity exec /work/02129/jayab867/singularity_cache/apegen-singularity-v1.0.simg python /APE-Gen/APE_Gen.py " + sequence + " " + hla + ".pdb " + additional_flags + " > log.txt\n")
    else:
      print("Error: hla pdb not found")
      sys.exit(0)
      #f.write("cd " + hla + "; for i in {0.." + str(num_replicates-1) + "}; do mkdir $i; cd $i; singularity exec /work/02129/jayab867/singularity_cache/apegen-singularity-v1.0.simg python /APE-Gen/APE_Gen.py " + sequence + " " + hla_name + " " + additional_flags + " > log.txt; cd ..; done; cd ..\n")
    #f.write("for i in {0.." + str(num_replicates-1) + "}; do mkdir $i; cd $i; cp ../../" + receptor_name + " .; singularity exec /work/02129/jayab867/singularity_cache/apegen-singularity-v1.0.simg python /APE-Gen/APE_Gen.py " + sequence + " " + receptor_name + " " + additional_flags + " > log.txt; cd ..; done\n")
    f.write("cd ../..\n")
    f.write("if test -f " + hla + "/" + sequence + "/0/min_energy_system.pdb; then mv " + hla + "/" + sequence + "/0 " + folder_location + "confs/" + hla + "-" + sequence + "; fi\n")
    f.write("rm -r " + hla + "/" + sequence + "\n")
    f.write("sleep 600\n\n")
    #f.close()

    #call(["less temp.txt >> run_sampling_long_" + str(num_scripts) + ".sh"], shell=True)
    counter += 1
    if counter == num_runs_per_script:
      f.close()
      counter = 0
      num_scripts += 1
      call(["cp run_sampling_long.sh run_sampling_long_" + str(num_scripts) + ".sh"], shell=True)
      f = open("run_sampling_long_" + str(num_scripts) + ".sh", 'a')
      #call(["sed -i \"s/test/test-" + str(num_scripts) + "/g\" run_sampling_long_" + str(num_scripts) + ".sh"], shell=True)
      #call(["sed -i \"s/log/log-" + str(num_scripts) + "/g\" run_sampling_long_" + str(num_scripts) + ".sh"], shell=True)
    

# make a jobfile and launcher.slurm for every 8 bash files
num_sh_files = len(glob.glob("run_sampling_long_*.sh"))

counter = 0
jobfile_counter = 0
while counter < num_sh_files:

  f = open("jobfile" + str(jobfile_counter), 'w')

  for i in range(64): 
    f.write("bash " + folder_location + "run_sampling_long_" + str(counter) + ".sh\n")
    counter += 1

  f.close()

  call(["cp launcher.slurm launcher" + str(jobfile_counter) + ".slurm"], shell=True)
  #call(["sed -i \"s/JOBNAME/test-" + str(jobfile_counter) + "/g\" launcher" + str(jobfile_counter) + ".slurm"], shell=True)
  call(["sed -i \"s/JOBFILE_LOCATION/jobfile" + str(jobfile_counter) + "/g\" launcher" + str(jobfile_counter) + ".slurm"], shell=True)

  jobfile_counter += 1






