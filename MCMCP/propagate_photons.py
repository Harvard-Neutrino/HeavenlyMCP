import argparse
import os, sys
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-i',dest='input',type=str,default="mcp_events.f2k",
                    help='Input energy losses in F2K format')
parser.add_argument('-o',dest='output',type=str,default="mcp_hits.ppc",
                    help='Output hits in PPC format.')
parser.add_argument('-d',dest='device',type=int,default=0,
                    help='Device to use in the calculation, GPU or CPU.')
parser.add_argument('--PPCTables',dest='ppctables',type=str,default="./PPC",
                    help='Location of the PPC tables.')
parser.add_argument('--PPCExe',dest='ppcexe',type=str,default="./PPC/ppc",
                    help='Location of the PPC executable.')

args = parser.parse_args()

command = f'{args.ppcexe} {args.device} < {args.input} > {args.output}'
print(command)

tenv = os.environ.copy()
tenv['PPCTABLESDIR'] = args.ppctables

process = subprocess.Popen(command,
                           shell=True,
                           stdout=subprocess.PIPE,
                           env=tenv
                          )
process.wait()