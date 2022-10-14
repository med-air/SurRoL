w_lines=[]
cnt=0
with open("") as f:
   lines = f.readlines()
   f_flag=True
   for line in lines:
       if(f_flag and line[0]=='v'):
           convex_line="o convex_"+str(cnt)+"\n"
           w_lines.append(convex_line)
           w_lines.append(line)
           cnt+=1
           f_flag=False
       else:
           w_lines.append(line)
           if (line[0]=='f'):
               f_flag=True
with open("","w") as fw:
   fw.writelines(w_lines)