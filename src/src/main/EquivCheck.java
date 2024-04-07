package main;
import com.alibaba.fastjson.JSONObject;
import main.Rewriter;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.rel2sql.RelToSqlConverter;
import org.apache.calcite.sql.SqlDialect;
import org.apache.commons.io.filefilter.FalseFileFilter;
import org.apache.commons.lang3.tuple.Pair;
import main.httputil;
import java.util.Vector;

public class EquivCheck {
    public static void checkeq(Rewriter rewriter, String sql1, String sql2, RelNode rel1, RelNode rel2){
        if( 1 == 1) {
            return ;
        }
        String Query = "";
        boolean odd_quote = Boolean.FALSE;
        int begin_quote = 0, end_quote;
        for(int i = 0; i < sql1.length(); ++ i){
            if(sql1.charAt(i) == '\''){
                if(!odd_quote) {
                    begin_quote = i + 1;
                    odd_quote = Boolean.TRUE;
                } else {
                    end_quote = i;
                    odd_quote = Boolean.FALSE;
                    String tmp = sql1.substring(begin_quote, end_quote);
                    sql2 = sql2.replace(tmp, "'" +tmp+ "'");
                }
            }
        }
        sql2 = sql2.replace("\"","");
        SqlDialect d = SqlDialect.DatabaseProduct.HIVE.getDialect();
        RelToSqlConverter converter = new RelToSqlConverter(d);
        sql1 = converter.visitRoot(rel1).asStatement().toSqlString(d).getSql();
        sql2 = converter.visitRoot(rel2).asStatement().toSqlString(d).getSql();
        JSONObject post = new JSONObject();
        post.put("api_key", "ffd63466745df3c65e4be7248eb22b92");
        for(int i = 0; i < rewriter.schema.size(); ++ i){
            Pair<String, Vector<Pair<String, String>>> sch_pair = rewriter.schema.get(i);
            String tableName = sch_pair.getLeft();
            Query += "schema sch_" + tableName + "(";
            boolean comma = Boolean.FALSE;
            for(Pair<String, String> col : sch_pair.getRight()) {
                String colName = col.getLeft();
                String colType = col.getRight();
                if(comma){
                    Query += ", ";
                } else { comma = true; }
                Query += colName+":"+colType;
            }
            Query += ");\n";
        }
        for(int i = 0; i < rewriter.schema.size(); ++ i){
            Pair<String, Vector<Pair<String, String>>> sch_pair = rewriter.schema.get(i);
            String tableName = sch_pair.getLeft();
            Query += "table " + tableName + "(" + "sch_" + tableName + ");\n";
        }
        Query += "query q1 \n";
        Query += "`" + sql1 + "`;\n";
        Query += "query q2 \n";
        Query += "`" + sql2 + "`;\n";
        Query += "verify q1 q2\n";
        post.put("query",Query);
        System.out.println("Query:"+Query);
    }
}
