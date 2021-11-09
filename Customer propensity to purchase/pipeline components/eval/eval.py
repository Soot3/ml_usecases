import argparse
def eval(lr_metrics, rf_metrics,gnb_metrics):
    import joblib
    import pandas as pd
    import pprint
    lr_metrics=joblib.load(lr_metrics)
    rf_metrics=joblib.load(rf_metrics)
    gnb_metrics=joblib.load(gnb_metrics)

    lis = [lr_metrics,rf_metrics,gnb_metrics]
    max = 0
    for i in lis:
        accuracy = i['test']
        if accuracy > max:
            max = accuracy
            model = i['model']
            metrics = i
  
    print('Best metrics \n\n')
    pprint.pprint(metrics)

    joblib.dump(model, 'best_model')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_metrics')
    parser.add_argument('--rf_metrics')
    parser.add_argument('--gnb_metrics')
    args = parser.parse_args()
    eval(args.lr_metrics,args.rf_metrics,args.gnb_metrics)