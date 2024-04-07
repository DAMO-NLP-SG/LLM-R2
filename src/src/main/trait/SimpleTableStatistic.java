package main.trait;

import org.apache.calcite.rel.RelCollation;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelReferentialConstraint;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.util.ImmutableBitSet;

import java.util.Collections;
import java.util.List;

public class SimpleTableStatistic implements Statistic {

    private final long rowCount;

    public SimpleTableStatistic(long rowCount) {
        this.rowCount = rowCount;
    }

    @Override
    public Double getRowCount() {
        return (double) rowCount;
    }

    @Override
    public boolean isKey(ImmutableBitSet columns) {
        return false;
    }

    @Override
    public List<ImmutableBitSet> getKeys() {
        return Collections.emptyList();
    }

    @Override
    public List<RelReferentialConstraint> getReferentialConstraints() {
        return Collections.emptyList();
    }

    @Override
    public List<RelCollation> getCollations() {
        return Collections.emptyList();
    }

    @Override
    public RelDistribution getDistribution() {
        return RelDistributionTraitDef.INSTANCE.getDefault();
    }
}
