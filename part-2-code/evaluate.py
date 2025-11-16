from argparse import ArgumentParser
from utils import compute_metrics, read_queries

parser = ArgumentParser()
parser.add_argument("-ps", "--predicted_sql", dest = "pred_sql",
    required = True, help = "path to your model's predicted SQL queries")
parser.add_argument("-pr", "--predicted_records", dest = "pred_records",
    required = True, help = "path to the predicted development database records")
parser.add_argument("-ds", "--development_sql", dest = "dev_sql",
    required = True, help = "path to the ground-truth development SQL queries")
parser.add_argument("-dr", "--development_records", dest = "dev_records",
    required = True, help = "path to the ground-truth development database records")

args = parser.parse_args()
sql_em, record_em, record_f1, model_error_msgs = compute_metrics(args.dev_sql, args.pred_sql, args.dev_records, args.pred_records)
print("SQL EM: ", sql_em)
print("Record EM: ", record_em)
print("Record F1: ", record_f1)
# print("Model Error Messages: ", model_error_msgs)

qs = read_queries(args.pred_sql)



syntax = 0
incomp = 0
no_col = 0
unrec_tok = 0
amb = 0
time_out = 0
syntax_f = None
incomp_f = None
no_col_f = None
unrec_tok_f = None
amb_f = None
time_out_f = None
for i in range(len(model_error_msgs)):
    msg = model_error_msgs[i].lower()
    if 'syntax error' in msg:
        syntax += 1
        if syntax_f is None:
            syntax_f = qs[i]
        elif len(syntax_f) > qs[i].__len__():
            syntax_f = qs[i]
    elif 'incomplete input' in msg:
        incomp += 1
        if incomp_f is None:
            incomp_f = qs[i]
        elif len(incomp_f) > qs[i].__len__():
            incomp_f = qs[i]
    elif 'no such column' in msg:
        no_col += 1
        if no_col_f is None:
            no_col_f = qs[i]
        elif len(no_col_f) > qs[i].__len__():
            no_col_f = qs[i]
    elif 'unrecognized token' in msg:
        unrec_tok += 1
        if unrec_tok_f is None:
            unrec_tok_f = qs[i]
        elif len(unrec_tok_f) > qs[i].__len__():
            unrec_tok_f = qs[i]
    elif 'ambiguous column name' in msg:
        amb += 1
        if amb_f is None:
            amb_f = qs[i]
        elif len(amb_f) > qs[i].__len__():
            amb_f = qs[i]
    elif 'query timed out' in msg:
        time_out += 1
        if time_out_f is None:
            time_out_f = qs[i]
        elif len(time_out_f) > qs[i].__len__():
            time_out_f = qs[i]

print(syntax_f)
print(incomp_f)
print(no_col_f)
print(unrec_tok_f)
print(amb_f)
print(time_out_f)


total = len(model_error_msgs)
# print(model_error_msgs)
# print(f'{total} queries in total.')
# print(f'Syntax Error: {syntax/total*100:.2f}%')
# print(f'Incomplete Input: {incomp/total*100:.2f}%')
# print(f'Unknown Column: {no_col/total*100:.2f}%')
# print(f'Unrecognized Token: {unrec_tok/total*100:.2f}%')
# print(f'Ambiguous Column Name: {amb/total*100:.2f}%')
# print(f'Query Time Out: {time_out/total*100:.2f}%')
print(f'{total} queries in total.')
print(f'Syntax Error: {syntax}/{total}')
print(f'Incomplete Input: {incomp}/{total}')
print(f'Unknown Column: {no_col}/{total}')
print(f'Unrecognized Token: {unrec_tok}/{total}')
print(f'Ambiguous Column Name: {amb}/{total}')
print(f'Query Time Out: {time_out}/{total}')