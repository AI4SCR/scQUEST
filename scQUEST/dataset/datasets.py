from ._utils import DataSet


def breastCancerAtlas(force_load=False):
    """Flow cytometry dataset with annotated epithelial cells in `['celltype', 'celltype_class']` of `ad.obs`. Locally stored at ~/.scQUEST"""
    return DataSet(
        name="breastCancerAtlas",
        url="https://figshare.com/ndownloader/files/34486574",
        doc_header="Processed breast cancer atlas from https://doi.org/10.1016/j.cell.2019.03.005",
        force_load=force_load,
    )()


def breastCancerAtlasRaw(force_load=False):
    """Raw flow cytometry dataset. Locally stored at ~/.scQUEST"""
    return DataSet(
        name="breastCancerAtlasRaw",
        url="https://figshare.com/ndownloader/files/34488644",
        doc_header="Raw breast cancer atlas from https://doi.org/10.1016/j.cell.2019.03.005",
        force_load=force_load,
    )()


def aml_and_healthy(force_load=False):
    """AML dataset with cells from healthy and AML patients. Locally stored at ~/.scQUEST"""
    return DataSet(
        name="aml_and_healthy",
        url="https://figshare.com/ndownloader/files/35283664",
        doc_header="Subset of Data from Data-driven phenotypic dissection of AML reveals progenitor-like cells that correlate with prognosis (10.1016/j.cell.2015.05.047). ",
        force_load=force_load,
    )()


def aml_annotated_celltypes(force_load=False):
    """AML dataset with cells from healthy donors to show case classification. Locally stored at ~/.scQUEST"""
    return DataSet(
        name="aml_annotated_celltypes",
        url="https://figshare.com/ndownloader/files/35284078",
        doc_header="Subset of Data from Data-driven phenotypic dissection of AML reveals progenitor-like cells that correlate with prognosis (10.1016/j.cell.2015.05.047). ",
        force_load=force_load,
    )()
