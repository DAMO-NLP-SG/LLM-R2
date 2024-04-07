package main.trait;

import org.apache.calcite.DataContext;
import org.apache.calcite.linq4j.Enumerable;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeFieldImpl;
import org.apache.calcite.rel.type.RelRecordType;
import org.apache.calcite.rel.type.StructKind;
import org.apache.calcite.schema.ScannableTable;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.impl.AbstractTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class SimpleTable extends AbstractTable implements ScannableTable {

    private final String tableName;
    private final List<String> fieldNames;
    private final List<SqlTypeName> fieldTypes;
    private final SimpleTableStatistic statistic;

    private RelDataType rowType;

    private SimpleTable(String tableName, List<String> fieldNames, List<SqlTypeName> fieldTypes, SimpleTableStatistic statistic) {
        this.tableName = tableName;
        this.fieldNames = fieldNames;
        this.fieldTypes = fieldTypes;
        this.statistic = statistic;
    }

    public String getTableName() {
        return tableName;
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
        if (rowType == null) {
            List<RelDataTypeField> fields = new ArrayList<>(fieldNames.size());

            for (int i = 0; i < fieldNames.size(); i++) {
                RelDataType fieldType = typeFactory.createSqlType(fieldTypes.get(i));
                RelDataTypeField field = new RelDataTypeFieldImpl(fieldNames.get(i), i, fieldType);
                fields.add(field);
            }

            rowType = new RelRecordType(StructKind.PEEK_FIELDS, fields, false);
        }

        return rowType;
    }

    @Override
    public Statistic getStatistic() {
        return statistic;
    }

    @Override
    public Enumerable<Object[]> scan(DataContext root) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static Builder newBuilder(String tableName) {
        return new Builder(tableName);
    }

    public static final class Builder {

        private final String tableName;
        private final List<String> fieldNames = new ArrayList<>();
        private final List<SqlTypeName> fieldTypes = new ArrayList<>();
        private long rowCount;

        private Builder(String tableName) {
            if (tableName == null || tableName.isEmpty()) {
                throw new IllegalArgumentException("Table name cannot be null or empty");
            }

            this.tableName = tableName;
        }

        public Builder addField(String name, SqlTypeName typeName) {
            if (name == null || name.isEmpty()) {
                throw new IllegalArgumentException("Field name cannot be null or empty");
            }

            if (fieldNames.contains(name)) {
                throw new IllegalArgumentException("Field already defined: " + name);
            }

            fieldNames.add(name);
            fieldTypes.add(typeName);

            return this;
        }

        public Builder withRowCount(long rowCount) {
            this.rowCount = rowCount;

            return this;
        }

        public SimpleTable build() {
            if (fieldNames.isEmpty()) {
                throw new IllegalStateException("Table must have at least one field");
            }

//            if (rowCount == 0L) {
//                throw new IllegalStateException("Table must have positive row count");
//            }

            return new SimpleTable(tableName, fieldNames, fieldTypes, new SimpleTableStatistic(rowCount));
        }

        public Pair<String, Vector<Pair<String, String>>> tableDiscribe(){
            Vector<Pair<String, String>> v = new Vector<>();
            for(int i = 0;i < fieldNames.size(); ++ i){
                String fieldName = fieldNames.get(i);
                String fieldType = fieldTypes.get(i).getName();
                v.add(Pair.of(fieldName, fieldType));
            }
            return Pair.of(tableName, v);
        }
    }
}
