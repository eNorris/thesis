#include "treemodel.h"

#include "ampxparser.h"
#include "treeitem.h"

#include <QDebug>

TreeModel::TreeModel(const AmpxParser &parser, QObject *parent)
    : QAbstractItemModel(parent)
{
    QList<QVariant> rootData;
    rootData << "Record" << "Data";
    rootItem = new TreeItem(rootData);
    linkModelData(parser);
}

TreeModel::~TreeModel()
{
    delete rootItem;
}

QModelIndex TreeModel::index(int row, int column, const QModelIndex &parent)
            const
{
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    TreeItem *parentItem;

    if (!parent.isValid())
        parentItem = rootItem;
    else
        parentItem = static_cast<TreeItem*>(parent.internalPointer());

    TreeItem *childItem = parentItem->child(row);
    if (childItem)
        return createIndex(row, column, childItem);
    else
        return QModelIndex();
}

QModelIndex TreeModel::parent(const QModelIndex &index) const
{
    if (!index.isValid())
        return QModelIndex();

    TreeItem *childItem = static_cast<TreeItem*>(index.internalPointer());
    TreeItem *parentItem = childItem->parentItem();

    if (parentItem == rootItem)
        return QModelIndex();

    return createIndex(parentItem->row(), 0, parentItem);
}

int TreeModel::rowCount(const QModelIndex &parent) const
{
    TreeItem *parentItem;
    if (parent.column() > 0)
        return 0;

    if (!parent.isValid())
        parentItem = rootItem;
    else
        parentItem = static_cast<TreeItem*>(parent.internalPointer());

    return parentItem->childCount();
}

int TreeModel::columnCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return static_cast<TreeItem*>(parent.internalPointer())->columnCount();
    else
        return rootItem->columnCount();
}

QVariant TreeModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if (role != Qt::DisplayRole)
        return QVariant();

    TreeItem *item = static_cast<TreeItem*>(index.internalPointer());

    return item->data(index.column());
}

Qt::ItemFlags TreeModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;

    return QAbstractItemModel::flags(index);
}

QVariant TreeModel::headerData(int section, Qt::Orientation orientation,
                               int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
        return rootItem->data(section);

    return QVariant();
}

void TreeModel::linkModelData(const AmpxParser &parser)
{
    //QList<TreeItem*> parents;
    //QList<int> indentations;
    //parents << rootItem;  //parent;
    //indentations << 0;

    //parents.last()->appendChild(new TreeItem(columnData, parents.last()));

    m_parser = &parser;

    //int number = 0;

    /*
    while (number < lines.count()) {
        int position = 0;
        while (position < lines[number].length()) {
            if (lines[number].mid(position, 1) != " ")
                break;
            position++;
        }

        QString lineData = lines[number].mid(position).trimmed();

        if (!lineData.isEmpty()) {
            // Read the column data from the rest of the line.
            QStringList columnStrings = lineData.split("\t", QString::SkipEmptyParts);
            QList<QVariant> columnData;
            for (int column = 0; column < columnStrings.count(); ++column)
                columnData << columnStrings[column];

            if (position > indentations.last()) {
                // The last child of the current parent is now the new parent
                // unless the current parent has no children.

                if (parents.last()->childCount() > 0) {
                    parents << parents.last()->child(parents.last()->childCount()-1);
                    indentations << position;
                }
            } else {
                while (position < indentations.last() && parents.count() > 0) {
                    parents.pop_back();
                    indentations.pop_back();
                }
            }

            // Append a new item to the current parent's list of children.
            parents.last()->appendChild(new TreeItem(columnData, parents.last()));
        }

        ++number;
    }
    */
}

void TreeModel::setupHeaderData()
{
    // Neutron Energy Structure
    QList<QVariant> nGroups;
    QList<QVariant> nStruct;
    nGroups << "Neutron Energy Groups";
    nGroups << m_parser->getNeutronEnergyGroups();
    nStruct << "Neutron Groups";
    QString myString;
    for (unsigned int i = 0; i < m_parser->getNeutronEnergy().size(); ++i)
        myString.append(QString::number(m_parser->getNeutronEnergy()[i]) + "  ");
    nStruct << myString;
    TreeItem *nTree = new TreeItem(nGroups, rootItem);
    nTree->appendChild(new TreeItem(nStruct, nTree));
    rootItem->appendChild(nTree);

    // Gamma Energy Structure
    QList<QVariant> gGroups;
    QList<QVariant> gStruct;
    gGroups << "Gamma Energy Groups";
    gGroups << m_parser->getGammaEnergyGroups();
    gStruct << "Gamma Groups";
    myString = "";
    for (unsigned int i = 0; i < m_parser->getGammaEnergy().size(); ++i)
        myString.append(QString::number(m_parser->getGammaEnergy()[i]) + "  ");
    gStruct << myString;
    TreeItem *gTree = new TreeItem(gGroups, rootItem);
    gTree->appendChild(new TreeItem(gStruct, gTree));
    rootItem->appendChild(gTree);

    // Nuclides
    for(int i = 0; i < m_parser->getNumberNuclides(); i++)
    {
        //qDebug() << "nuclide";
        QList<QVariant> nuclide;
        QList<QVariant> bondarenko;
        QList<QVariant> nXs;
        QList<QVariant> nScat;
        QList<QVariant> gXs;
        QList<QVariant> gProd;
        QList<QVariant> gScat;

        //qDebug() << m_parser->getDirectoryEntry(i)->getText();
        nuclide << QString::number(m_parser->getDirectoryEntry(i)->getId()) << m_parser->getDirectoryEntry(i)->getText();
        bondarenko << "Bondarenko" << "--";
        nXs << "Neutron XS" << "--";
        nScat << "Neutron Scatter" << "--";
        gXs << "Gamma XS" << "--";
        gProd << "Gamma Production" << "--";
        gScat << "Gamma Scatter" << "--";

        TreeItem *nuclideTree = new TreeItem(nuclide, rootItem);
        TreeItem *bondTree = new TreeItem(bondarenko, nuclideTree);
        TreeItem *nXsTree = new TreeItem(nXs, nuclideTree);
        TreeItem *nScatTree = new TreeItem(nScat, nuclideTree);
        TreeItem *gXsTree = new TreeItem(gXs, nuclideTree);
        TreeItem *gProdTree = new TreeItem(gProd, nuclideTree);
        TreeItem *gScatTree = new TreeItem(gScat, nuclideTree);

        nuclideTree->appendChild(bondTree);
        nuclideTree->appendChild(nXsTree);
        nuclideTree->appendChild(nScatTree);
        nuclideTree->appendChild(gXsTree);
        nuclideTree->appendChild(gProdTree);
        nuclideTree->appendChild(gScatTree);
        rootItem->appendChild(nuclideTree);
    }

    /*
    while (number < lines.count()) {
        int position = 0;
        while (position < lines[number].length()) {
            if (lines[number].mid(position, 1) != " ")
                break;
            position++;
        }

        QString lineData = lines[number].mid(position).trimmed();

        if (!lineData.isEmpty()) {
            // Read the column data from the rest of the line.
            QStringList columnStrings = lineData.split("\t", QString::SkipEmptyParts);
            QList<QVariant> columnData;
            for (int column = 0; column < columnStrings.count(); ++column)
                columnData << columnStrings[column];

            if (position > indentations.last()) {
                // The last child of the current parent is now the new parent
                // unless the current parent has no children.

                if (parents.last()->childCount() > 0) {
                    parents << parents.last()->child(parents.last()->childCount()-1);
                    indentations << position;
                }
            } else {
                while (position < indentations.last() && parents.count() > 0) {
                    parents.pop_back();
                    indentations.pop_back();
                }
            }

            // Append a new item to the current parent's list of children.
            parents.last()->appendChild(new TreeItem(columnData, parents.last()));
        }

        ++number;
    }
    */
}

void TreeModel::setupNuclideData()
{

}

void TreeModel::addFile(QString str)
{
    QList<QVariant> fDir;
    fDir << "File";
    fDir << str;
    rootItem->appendChild(new TreeItem(fDir, rootItem));
}
