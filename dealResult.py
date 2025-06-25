import os
import glob
import xlwt

result_summ_file = '...../project/result_output/addEmbed_sum/'
result_file = '...../project/result_output/'
def dealAllResult(roofile):
    outfile_num = '0'
    if roofile != 'addEmbed':
        if roofile == 'classnumAna':
            outfile_num = 'cs'
        else:
            outfile_num = str(roofile.split('_')[1])
    files = glob.glob(result_file+roofile + '/*')
    files = sorted(files)
    for file in files:
        f1_macro_count = 0
        f1_micro_count = 0
        auc_count = 0
        acc_count = 0
        f1_macro_value = 0
        f1_micro_value = 0
        auc_value = 0
        acc_value = 0
        f1_macro_max = 0
        f1_micro_max = 0
        auc_max = 0
        acc_max = 0
        f1_macro_min = 10
        f1_micro_min = 10
        auc_min = 10
        acc_min = 10
        with open(file, 'r',encoding='utf-8') as f:
            for line in f:
                if 'BestF1-macro:' in line:
                    f1_macro_count = f1_macro_count + 1
                    f1_macro_value = f1_macro_value + float(line.split(':')[1])
                    if float(line.split(':')[1]) > f1_macro_max:
                        f1_macro_max = float(line.split(':')[1])
                    if float(line.split(':')[1]) < f1_macro_min:
                        f1_macro_min = float(line.split(':')[1])
                elif 'BestF1-micro:' in line:
                    f1_micro_count = f1_micro_count + 1
                    f1_micro_value = f1_micro_value + float(line.split(':')[1])
                    if float(line.split(':')[1]) > f1_micro_max:
                        f1_micro_max = float(line.split(':')[1])
                    if float(line.split(':')[1]) < f1_micro_min:
                        f1_micro_min = float(line.split(':')[1])
                elif 'AUC:' in line:
                    auc_count = auc_count + 1
                    auc_value = auc_value + float(line.split(':')[1])
                    if float(line.split(':')[1]) > auc_max:
                        auc_max = float(line.split(':')[1])
                    if float(line.split(':')[1]) < auc_min:
                        auc_min = float(line.split(':')[1])
                elif 'ACC:' in line:
                    acc_count = acc_count + 1
                    acc_value = acc_value + float(line.split(':')[1])
                    if float(line.split(':')[1]) > acc_max:
                        acc_max = float(line.split(':')[1])
                    if float(line.split(':')[1]) < acc_min:
                        acc_min = float(line.split(':')[1])


        if not os.path.exists(result_summ_file+'result_sum'+outfile_num+'.txt'):
            os.mknod(result_summ_file+'result_sum'+outfile_num+'.txt', 0o666)
        with open(result_summ_file+'result_sum'+outfile_num+'.txt', 'a',encoding='utf-8') as w:
            w.write(file.split('/')[-1].split('.')[0]+':avg_f1_macro-'+str(f1_macro_value/f1_macro_count)+',f1_macro_max-'+str(f1_macro_max)+',f1_macro_min-'+str(f1_macro_min)+',middle-'+str((f1_macro_max+f1_macro_min)/2)+'+-'+str((f1_macro_max-f1_macro_min)/2)+'\n')
            w.write(
                file.split('/')[-1].split('.')[0] + ':avg_f1_micro-' + str(f1_micro_value / f1_micro_count) + ',f1_micro_max-' + str(
                    f1_micro_max) + ',f1_micro_min-' + str(f1_micro_min) + ',middle-' + str(
                    (f1_micro_max + f1_micro_min) / 2) + '+-' + str((f1_micro_max - f1_micro_min) / 2) + '\n')
            w.write(
                file.split('/')[-1].split('.')[0] + ':avg_auc-' + str(auc_value / auc_count) + ',auc_max-' + str(
                    auc_max) + ',auc_min-' + str(auc_min) + ',middle-' + str(
                    (auc_max + auc_min) / 2) + '+-' + str((auc_max - auc_min) / 2) + '\n')
            if acc_value != 0 :
                w.write(
                    file.split('/')[-1].split('.')[0] + ':avg_acc-' + str(acc_value / acc_count) + ',acc_max-' + str(
                        acc_max) + ',acc_min-' + str(acc_min) + ',middle-' + str(
                        (acc_max + acc_min) / 2) + '+-' + str((acc_max - acc_min) / 2) + '\n')

def dealk(datafile,resultfile,xlsfile):
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet('data')
    worksheet.write(0, 0, 'k')
    worksheet.write(0, 1, 'f1-macro')
    worksheet.write(0, 2, 'f1-micro/acc')
    worksheet.write(0, 3, 'auc')

    f1_macro_count = {}
    f1_micro_count = {}
    auc_count = {}
    acc_count = {}
    f1_macro_value = {}
    f1_micro_value = {}
    auc_value = {}
    acc_value = {}
    f1_macro_max = {}
    f1_micro_max = {}
    auc_max = {}
    acc_max = {}
    f1_macro_min = {}
    f1_micro_min = {}
    auc_min = {}
    acc_min = {}
    with open(datafile,'r',encoding='utf-8') as f:
        runnum = 0
        for line in f:
            if 'class-2-run' in line:
                if runnum != int(line.split(':')[1]):
                    runnum = int(line.split(':')[1])
                    f1_macro_count[str(runnum)] = 0
                    f1_micro_count[str(runnum)] = 0
                    auc_count[str(runnum)] = 0
                    acc_count[str(runnum)] = 0
                    f1_macro_max[str(runnum)] = 0
                    f1_micro_max[str(runnum)] = 0
                    auc_max[str(runnum)] = 0
                    acc_max[str(runnum)] = 0
                    f1_macro_min[str(runnum)] = 10
                    f1_micro_min[str(runnum)] = 10
                    auc_min[str(runnum)] = 10
                    acc_min[str(runnum)] = 10
                    f1_macro_value[str(runnum)] = 0
                    f1_micro_value[str(runnum)] = 0
                    auc_value[str(runnum)] = 0
                    acc_value[str(runnum)] = 0
            if 'BestF1-macro:' in line:
                f1_macro_count[str(runnum)] = f1_macro_count[str(runnum)] + 1
                f1_macro_value[str(runnum)] = f1_macro_value[str(runnum)] + float(line.split(':')[1])
                if float(line.split(':')[1]) > f1_macro_max[str(runnum)]:
                    f1_macro_max[str(runnum)] = float(line.split(':')[1])
                if float(line.split(':')[1]) < f1_macro_min[str(runnum)]:
                    f1_macro_min[str(runnum)] = float(line.split(':')[1])
            elif 'BestF1-micro:' in line:
                f1_micro_count[str(runnum)] = f1_micro_count[str(runnum)] + 1
                f1_micro_value[str(runnum)] = f1_micro_value[str(runnum)] + float(line.split(':')[1])
                if float(line.split(':')[1]) > f1_micro_max[str(runnum)]:
                    f1_micro_max[str(runnum)] = float(line.split(':')[1])
                if float(line.split(':')[1]) < f1_micro_min[str(runnum)]:
                    f1_micro_min[str(runnum)] = float(line.split(':')[1])
            elif 'AUC:' in line:
                auc_count[str(runnum)] = auc_count[str(runnum)] + 1
                auc_value[str(runnum)] = auc_value[str(runnum)] + float(line.split(':')[1])
                if float(line.split(':')[1]) > auc_max[str(runnum)]:
                    auc_max[str(runnum)] = float(line.split(':')[1])
                if float(line.split(':')[1]) < auc_min[str(runnum)]:
                    auc_min[str(runnum)] = float(line.split(':')[1])
            elif 'ACC:' in line:
                acc_count[str(runnum)] = acc_count[str(runnum)] + 1
                acc_value[str(runnum)] = acc_value[str(runnum)] + float(line.split(':')[1])
                if float(line.split(':')[1]) > acc_max[str(runnum)]:
                    acc_max[str(runnum)] = float(line.split(':')[1])
                if float(line.split(':')[1]) < acc_min[str(runnum)]:
                    acc_min[str(runnum)] = float(line.split(':')[1])

    if not os.path.exists(resultfile):
        os.mknod(resultfile, 0o666)
    for k in range(2,21):
        worksheet.write(k-1, 0, k)
        worksheet.write(k-1, 1, f1_macro_value[str(k)] / f1_macro_count[str(k)])
        worksheet.write(k-1, 2, f1_micro_value[str(k)] / f1_micro_count[str(k)])
        worksheet.write(k-1, 3, auc_value[str(k)] / auc_count[str(k)])
        with open(resultfile, 'a', encoding='utf-8') as w:
            w.write('k-'+str(k) + ':avg_f1_macro-' + str(
                f1_macro_value[str(k)] / f1_macro_count[str(k)]) + ',f1_macro_max-' + str(f1_macro_max[str(k)]) + ',f1_macro_min-' + str(
                f1_macro_min[str(k)]) + ',middle-' + str((f1_macro_max[str(k)] + f1_macro_min[str(k)]) / 2) + '+-' + str(
                (f1_macro_max[str(k)] - f1_macro_min[str(k)]) / 2) + '\n')
            w.write(
                'k'+str(k) + ':avg_f1_micro-' + str(
                    f1_micro_value[str(k)] / f1_micro_count[str(k)]) + ',f1_micro_max-' + str(
                    f1_micro_max[str(k)]) + ',f1_micro_min-' + str(f1_micro_min[str(k)]) + ',middle-' + str(
                    (f1_micro_max[str(k)] + f1_micro_min[str(k)]) / 2) + '+-' + str((f1_micro_max[str(k)] - f1_micro_min[str(k)]) / 2) + '\n')
            w.write(
                'k'+str(k)  + ':avg_auc-' + str(auc_value[str(k)] / auc_count[str(k)]) + ',auc_max-' + str(
                    auc_max[str(k)]) + ',auc_min-' + str(auc_min[str(k)]) + ',middle-' + str(
                    (auc_max[str(k)] + auc_min[str(k)]) / 2) + '+-' + str((auc_max[str(k)] - auc_min[str(k)]) / 2) + '\n')
            if acc_value != 0:
                w.write(
                    'k'+str(k)  + ':avg_acc-' + str(acc_value[str(k)] / acc_count[str(k)]) + ',acc_max-' + str(
                        acc_max[str(k)]) + ',acc_min-' + str(acc_min[str(k)]) + ',middle-' + str(
                        (acc_max[str(k)] + acc_min[str(k)]) / 2) + '+-' + str((acc_max[str(k)] - acc_min[str(k)]) / 2) + '\n')

    workbook.save(xlsfile)



def dealData():
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet('data')
    worksheet.write(0, 0, 'alpha')
    worksheet.write(0, 1, 'beta')
    worksheet.write(0, 2, 'gamma')
    worksheet.write(0, 3, 'value')
    values = 0
    num = 1
    with open( '...../project/result_output/addEmbed_23/result_chameleon_HGT_conv_diffusion.txt', 'r', encoding='utf-8') as w:
        for line in w:
            if 'BestF1-macro' in line and 'epoch' not in line:
                values = float(line.split(':')[1])
            if 'alpha' in line:
                worksheet.write(num, 0, float(line.split('beta')[0].split(':')[1].split(' ',1)[0]))
                worksheet.write(num, 1, float(line.split('gamma')[0].split('beta')[1].split(':')[1].split(' ', 1)[0]))
                worksheet.write(num, 2, float(line.split('gamma')[1].split(':')[1]))
                worksheet.write(num, 3, values)
                num += 1
    workbook.save('...../project/result_output/addEmbed_23/data.xls')

# dealAllResult('classnumAna')
dealk('...../result_output/addEmbed_26/result_squirrel_HGT_diffusion.txt','...../project/result_output/addEmbed_sum/analysisK_squirrel.txt','...../project/result_output/addEmbed_26/squirrel_data.xls')
