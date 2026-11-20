import argparse
import json
from pathlib import Path
from .results import json_write


def main():
    parser=argparse.ArgumentParser(description='Physical Rydberg Kagome RG+DMRG')
    commands=parser.add_subparsers(dest='command',required=True)
    spec=commands.add_parser('spec'); spec.add_argument('--output',required=True)
    audit=commands.add_parser('audit'); audit.add_argument('--output',required=True)
    audit.add_argument('--L',type=int,required=True); audit.add_argument('--count',type=int,default=128)
    case=commands.add_parser('case')
    case.add_argument('--output',required=True)
    case.add_argument('--L',type=int,default=4); case.add_argument('--sample',type=int,required=True)
    case.add_argument('--model',choices=['full','nn_nnn','nn'],required=True)
    case.add_argument('--delta',type=float,required=True); case.add_argument('--omega',type=float,required=True)
    case.add_argument('--chi',type=int,required=True)
    case.add_argument('--dimensions',type=int,nargs='+',default=[32,64,128,256])
    case.add_argument('--pairs',type=int,default=20000)
    case.add_argument('--max-sweeps',type=int,default=120)
    case.add_argument('--no-exact',action='store_true')
    case.add_argument('--fixed-production',action='store_true')
    args=parser.parse_args()
    if args.command=='spec':
        from .config import specification
        json_write(args.output,specification())
    elif args.command=='audit':
        from .audit import geometry_audit
        result=geometry_audit(args.L,args.count)
        json_write(args.output,result)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    else:
        from .campaign import run_case
        result=run_case(args.output,args.L,args.sample,args.model,args.delta,args.omega,args.chi,
                        args.dimensions,args.pairs,args.max_sweeps,not args.no_exact,args.fixed_production)
        print(json.dumps({k:result[k] for k in ['solver_pass','sampling_moment_check','rg_converged','seconds']},indent=2))


if __name__=='__main__': main()
