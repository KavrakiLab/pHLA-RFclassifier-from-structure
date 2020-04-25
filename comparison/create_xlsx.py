import sys
import xlsxwriter
from subprocess import call
import csv

methods = ["allele_ensemblesig", "allelesig", "allele_ensemblereg", "allelereg", "allele_ensembler2", "alleler2", "netMHCstabpan", "netMHCpan", "netmhc", "mhcflurry", "mixmhcpred"]
sheet_names = ["sig-ensemble", "sig-single", "recip-ensemble", "recip-single", "recip2-ensemble", "recip2-single", "netMHCstabpan", "netMHCpan", "netmhc", "mhcflurry", "mixmhcpred"]


workbook = xlsxwriter.Workbook("auroc_results.xlsx")
for i in range(len(methods)):
    worksheet_i = workbook.add_worksheet(sheet_names[i])
    worksheet_i.write(0, 0, "Allele")
    worksheet_i.write(0, 1, "AUROC")

    with open(methods[i] + ".csv", 'rt', encoding='utf8') as f:
        reader = csv.reader(f)
        for r, row in enumerate(reader):
            for c, col in enumerate(row):
                if c == 0: worksheet_i.write(r+1, 0, col)
                elif c == 3: worksheet_i.write_number(r+1, 1, float(col))

workbook.close()


