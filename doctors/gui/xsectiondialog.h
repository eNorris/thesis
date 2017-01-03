#ifndef XSECTIONDIALOG_H
#define XSECTIONDIALOG_H

#include <QDialog>

class AmpxParser;

class XSection;

namespace Ui {
class XSectionDialog;
}

class XSectionDialog : public QDialog
{
    Q_OBJECT

public:
    explicit XSectionDialog(QWidget *parent = 0);
    ~XSectionDialog();

    void updateXs(XSection *xs);
    void setXs(AmpxParser *p);

private:
    Ui::XSectionDialog *ui;
    XSection *m_xs;
};

#endif // XSECTIONDIALOG_H
