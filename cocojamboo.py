# This function is copied from COCOJumboo package at https://pypi.org/project/cocojamboo/, which now seems to be unavailable.
def coco_evaluator_class_metrics(model, test_loader, device):
    from engine import evaluate
    import numpy as np 
    coco_evaluator = evaluate(model, test_loader, device)
    for iou_type in coco_evaluator.iou_types:
        cocoEval = coco_evaluator.coco_eval[iou_type]
        catIds = cocoEval.params.catIds
        params  = cocoEval.params
        dict = cocoEval.eval
        iStrTable = '{}[IoU={IoUStr}]' 
        iStr = ' {:<18} {} @[ IoU={IoUStr:<9} | area={areaStr:>6s} | maxDets={maxDetsStr:>3d} | label={labelStr} ]'
        aind = [i for i, aRng in enumerate(params.areaRngLbl) if aRng == 'all']
        mind = [i for i, mDet in enumerate(params.maxDets) if mDet == 100]
        table, columns = [], []
        columns.append('Class')
        for cat in range(len(catIds)):
            item = []
            item.append(catIds[cat])
            for iouThr in [None, 0.50, 0.75]:
                precision = dict['precision']
                if iouThr is not None:
                    t = np.where(iouThr == params.iouThrs)[0]
                    precision = precision[t]
                precision = precision[:,:,:,aind,mind]
                if len(precision[precision>-1])==0:
                    mean_precision = -1
                else:
                    mean_precision = np.mean(precision[:,:,cat,:])
                    iouStr = '{:0.2f}:{:0.2f}'.format(params.iouThrs[0], params.iouThrs[-1]) \
                    if iouThr is None else '{:0.2f}'.format(iouThr)
                    info = iStr.format('Average Precision', 'AP', IoUStr=iouStr, areaStr='all', maxDetsStr=100, labelStr=cat)
                    info_table = iStrTable.format('AP', IoUStr=iouStr)
                    if info_table is None:
                        columns.append(info_table)
                    else:
                        if info_table not in columns:
                            columns.append(info_table)
                    item.append(mean_precision)
            iouThr = None
            recall = dict['recall']
            if iouThr is not None:
                t = np.where(iouThr == params.iouThrs)[0]
                recall = recall[t]
            recall = recall[:,:,aind,mind]
            if len(recall[recall>-1])==0:
                mean_recall = -1
            else:
                mean_recall = np.mean(recall[:,cat,:])
                iouStr = '{:0.2f}:{:0.2f}'.format(params.iouThrs[0], params.iouThrs[-1]) \
                    if iouThr is None else '{:0.2f}'.format(iouThr)
                info = iStr.format('Average Recall', 'AR', IoUStr=iouStr, areaStr='all', maxDetsStr=100, labelStr=cat)
                info_table = iStrTable.format('AR', IoUStr=iouStr)
                if info_table not in columns:
                    columns.append(info_table)
                item.append(mean_recall)    
            table.append(item)  
    return table, columns 
