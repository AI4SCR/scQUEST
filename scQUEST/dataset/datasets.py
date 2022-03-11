from ._utils import DataSet


def breastCancerAtlas():
    """Flow cytometry dataset with annotated epithelial cells"""
    return DataSet(
        name='breastCancerAtlas',
        url='https://figshare.com/ndownloader/files/33966143',
        doc_header='Processed breast cancer atlas from https://doi.org/10.1016/j.cell.2019.03.005')()


def breastCancerAtlasRaw():
    """Raw flow cytometry dataset"""
    return DataSet(
        name='breastCancerAtlasRaw',
        url='https://figshare.com/ndownloader/files/34036679',
        doc_header='Raw breast cancer atlas from https://doi.org/10.1016/j.cell.2019.03.005'
    )()
